import asyncio
import ctypes
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import cycle
from math import prod
from pathlib import Path

import orjson
import pybase64
import verifiers as vf
from verifiers.utils.client_utils import setup_openai_client
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.transport import TrainingSample
from prime_rl.transport.types import EncodedTensor
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import InterceptHandler, get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the student inference pool + matching renderer. Returns
    ``(renderer | None, inference_pool)``; ``renderer`` is ``None`` on the
    MITO path (``config.renderer is None``)."""
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name

    if config.renderer is not None:
        renderer = create_renderer(tokenizer, config.renderer)
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=config.renderer,
            pool_size=config.pool_size,
        )
        get_logger().info("Using direct renderer rollout client")
        return renderer, inference_pool

    get_logger().info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="openai_chat_completions",
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


def get_model_completion_len(output: vf.RolloutOutput) -> int:
    """Sum of model-generated completion tokens across all turns (excludes
    environment-injected tokens between turns)."""
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"] if step.get("tokens"))


def get_tool_response_len(output: vf.RolloutOutput) -> int:
    """Total tool-response tokens consumed across the whole rollout, read from a
    harness-emitted metric (e.g. RLM's `rlm_total_tool_response_tokens`, deduped
    across turns/branches/sub-RLMs). Returns 0 when no such metric is present."""
    metrics = output.get("metrics") or {}
    for key, value in metrics.items():
        if key.endswith("total_tool_response_tokens") and isinstance(value, (int, float)):
            return int(value)
    return 0


def save_rollouts(rollouts: list[vf.RolloutOutput], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts to a JSONL file using verifiers serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "wb") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            f.write(orjson.dumps(row, default=make_serializable, option=opts))


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().info(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def trim_process_memory() -> None:
    """Return freed heap pages to the OS on glibc systems."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as exc:
        get_logger().debug(f"malloc_trim(0) failed: {exc!r}")


@dataclass
class TeacherPrefillScores:
    """Teacher scores aligned to the supplied input-token positions."""

    logprobs: list[float]
    topk_token_ids: EncodedTensor | None = None
    topk_logprobs: EncodedTensor | None = None


def _decode_compact_tensor(
    payload: object,
    *,
    name: str,
    expected_dtype: str,
    expected_shape: list[int],
) -> bytes:
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher compact {name} must be an object.")
    dtype = payload.get("dtype")
    shape = payload.get("shape")
    encoded = payload.get("data")
    if dtype != expected_dtype:
        raise ValueError(f"Teacher compact {name} has dtype {dtype!r}; expected {expected_dtype!r}.")
    if (
        not isinstance(shape, list)
        or any(type(dimension) is not int for dimension in shape)
        or shape != expected_shape
    ):
        raise ValueError(f"Teacher compact {name} has shape {shape!r}; expected {expected_shape!r}.")
    if not isinstance(encoded, str):
        raise ValueError(f"Teacher compact {name} data must be a base64 string.")
    try:
        decoded = pybase64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError(f"Teacher compact {name} contains invalid base64 data.") from error
    expected_nbytes = prod(expected_shape) * 4
    if len(decoded) != expected_nbytes:
        raise ValueError(
            f"Teacher compact {name} decoded to {len(decoded)} bytes; expected {expected_nbytes}."
        )
    return decoded


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
    top_k: int | None = None,
) -> list[TeacherPrefillScores]:
    """Prefill-score samples under the teacher.

    ``top_k`` requests the teacher's true top-k support at every position.
    vLLM also returns the realized input token when it falls outside that
    support. It is retained separately in ``logprobs`` so the trainer can
    form the teacher-top-k union sampled-token objective without changing the
    fixed-width sparse top-k tensors.
    """
    import httpx
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    if not samples:
        return []
    if not clients:
        raise ValueError("Teacher scoring requires at least one client.")
    teacher_clients = [setup_openai_client(client_config) for client_config in clients]

    async def _compute_single(client, sample: TrainingSample) -> TeacherPrefillScores:

        # Two escape hatches from ``AsyncOpenAI.post``:
        #   1. URL — ``/inference/v1/generate`` is mounted at server root, not
        #      under ``/v1``. Pass an absolute URL so the SDK's
        #      ``_prepare_url`` skips the base-url merge (it short-circuits
        #      when the path passes ``httpx.URL.is_relative_url`` as False).
        #   2. Parse — vLLM's ``GenerateResponse`` is a plain
        #      ``pydantic.BaseModel`` and the SDK's parse layer rejects any
        #      ``cast_to`` that doesn't subclass ``openai.BaseModel``. Use
        #      ``cast_to=httpx.Response`` so the SDK still builds the request
        #      (preserving ``auth_headers``, retries, timeouts, idempotency
        #      keys) and just hands us the raw response to validate ourselves.
        base = str(client.base_url).rstrip("/").removesuffix("/v1")
        sampling_params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            # Teacher scoring only consumes token ids and logprobs. Avoid
            # constructing decoded strings for every sparse support item.
            "detokenize": False,
            "prompt_logprobs": top_k if top_k is not None else 1,
        }
        body = {
            "model": model_name,
            "token_ids": list(sample.prompt_ids) + list(sample.completion_ids),
            "sampling_params": sampling_params,
        }
        if top_k is not None:
            # vLLM keeps [realized token, true top-k...] in primitive flat
            # arrays. Prime-RL's opt-in response transports those arrays
            # directly; a vLLM 0.22 server without the Prime-RL extension
            # ignores the marker and returns expanded JSON parsed below.
            sampling_params["flat_logprobs"] = True
            body["prime_rl_compact_prompt_logprobs"] = True

        http_response = await client.post(
            f"{base}/inference/v1/generate",
            cast_to=httpx.Response,
            body=body,
        )
        expected_len = len(sample.prompt_ids) + len(sample.completion_ids)

        # Detect the object-valued compact field without first parsing a
        # possible legacy multi-gigabyte JSON response. Whitespace is allowed
        # around the colon for test servers and non-default JSON renderers.
        marker = b'"prompt_logprobs_compact"'
        marker_index = http_response.content.find(marker)
        compact_response = False
        if marker_index >= 0:
            cursor = marker_index + len(marker)
            while cursor < len(http_response.content) and http_response.content[cursor] in b" \t\r\n:":
                cursor += 1
            compact_response = cursor < len(http_response.content) and http_response.content[cursor] == ord("{")

        if compact_response:
            payload = orjson.loads(http_response.content)
            compact = payload.get("prompt_logprobs_compact")
            if not isinstance(compact, dict):
                raise ValueError("Teacher compact prompt-logprob response is malformed.")
            sampled_data = _decode_compact_tensor(
                compact.get("sampled_logprobs"),
                name="sampled_logprobs",
                expected_dtype="float32",
                expected_shape=[expected_len],
            )
            assert top_k is not None
            topk_shape = [expected_len, top_k]
            topk_ids_data = _decode_compact_tensor(
                compact.get("topk_token_ids"),
                name="topk_token_ids",
                expected_dtype="int32",
                expected_shape=topk_shape,
            )
            topk_logprobs_data = _decode_compact_tensor(
                compact.get("topk_logprobs"),
                name="topk_logprobs",
                expected_dtype="float32",
                expected_shape=topk_shape,
            )

            import numpy as np

            return TeacherPrefillScores(
                logprobs=np.frombuffer(sampled_data, dtype=np.float32).tolist(),
                topk_token_ids=EncodedTensor(dtype="int32", shape=topk_shape, data=topk_ids_data),
                topk_logprobs=EncodedTensor(dtype="float32", shape=topk_shape, data=topk_logprobs_data),
            )

        response = GenerateResponse.model_validate_json(http_response.content)
        # ``prompt_logprobs[i]`` is a ``{token_id: Logprob}`` dict for tokens
        # the engine could score, or ``None`` for the leading token which has
        # no preceding context. Flatten to ``list[float]`` with 0.0 in the
        # unscored slot.
        rows = response.prompt_logprobs or []
        if len(rows) != expected_len:
            raise ValueError(f"Teacher returned {len(rows)} score rows for {expected_len} input tokens.")

        flat = [0.0] * expected_len
        topk_ids: list[list[int]] = []
        topk_logprobs: list[list[float]] = []

        def _logprob(value) -> float | None:
            return value.logprob if hasattr(value, "logprob") else value.get("logprob")

        def _sortable_logprob(value) -> float:
            logprob = _logprob(value)
            return float(logprob) if logprob is not None else float("-inf")

        input_ids = sample.prompt_ids + sample.completion_ids
        for position, entry in enumerate(rows):
            if not entry:
                if top_k is not None:
                    topk_ids.append([0] * top_k)
                    topk_logprobs.append([-1e9] * top_k)
                continue
            realized_token_id = input_ids[position]
            realized = entry.get(realized_token_id)
            if realized is None:
                realized = entry.get(str(realized_token_id))
            if realized is None:
                raise ValueError(
                    f"Teacher prompt-logprob row {position} does not contain realized token {realized_token_id}."
                )
            lp = _logprob(realized)
            flat[position] = float(lp) if lp is not None else 0.0
            if top_k is not None:
                ranked = sorted(entry.items(), key=lambda item: _sortable_logprob(item[1]), reverse=True)[:top_k]
                if len(ranked) != top_k:
                    raise ValueError(
                        f"Teacher returned only {len(ranked)} logprobs for a requested top-k of {top_k}. "
                        "Set the teacher inference server's max_logprobs to at least opd_top_k."
                    )
                topk_ids.append([int(token_id) for token_id, _ in ranked])
                topk_logprobs.append([_sortable_logprob(value) for _, value in ranked])

        if top_k is None:
            return TeacherPrefillScores(logprobs=flat)

        import numpy as np

        return TeacherPrefillScores(
            logprobs=flat,
            topk_token_ids=EncodedTensor.from_numpy(np.asarray(topk_ids, dtype=np.int32)),
            topk_logprobs=EncodedTensor.from_numpy(np.asarray(topk_logprobs, dtype=np.float32)),
        )

    try:
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(_compute_single(client, sample))
                for client, sample in zip(cycle(teacher_clients), samples)
            ]
        return [task.result() for task in tasks]
    finally:
        await asyncio.gather(*(client.close() for client in teacher_clients), return_exceptions=True)


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
