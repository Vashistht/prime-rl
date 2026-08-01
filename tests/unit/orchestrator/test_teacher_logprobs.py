import asyncio
import json

import httpx
import numpy as np
import pybase64
import verifiers as vf

from prime_rl.orchestrator import utils as orchestrator_utils
from prime_rl.transport import TrainingSample


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict):
        # Match what AsyncOpenAI exposes — utils.py reads ``str(client.base_url)``.
        self.base_url = "http://fake-host:8000/v1"
        self._payload = payload
        self.calls: list[dict] = []
        self.closed = False

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=200,
            content=json.dumps(self._payload).encode(),
            request=request,
        )

    async def close(self):
        self.closed = True


def test_compute_teacher_logprobs_empty_samples_does_not_create_clients(monkeypatch):
    def _unexpected_client(_):
        raise AssertionError("empty teacher batch must not create an HTTP client")

    monkeypatch.setattr(orchestrator_utils, "setup_openai_client", _unexpected_client)
    result = asyncio.run(
        orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[],
            top_k=64,
        )
    )
    assert result == []


def test_compute_teacher_logprobs_uses_inference_generate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [None, {"2": {"logprob": -0.7}}, {"3": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)

        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
        )

        assert result[0].logprobs == [0.0, -0.7, -0.3]
        assert result[0].topk_token_ids is None
        assert result[0].topk_logprobs is None
        assert fake_client.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
                "body": {
                    "model": "teacher-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "detokenize": False,
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]
        assert fake_client.closed is True

    asyncio.run(_run())


def test_compute_teacher_topk_separates_realized_token_outside_support(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-topk",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {
                        # Do not rely on response mapping order to locate the
                        # realized token or reconstruct the ranked support.
                        "9": {"logprob": -0.1, "rank": 1},
                        "2": {"logprob": -4.0, "rank": 3},
                        "8": {"logprob": -0.2, "rank": 2},
                    },
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)
        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[False],
            completion_ids=[2],
            completion_mask=[True],
            completion_logprobs=[-0.5],
            completion_temperatures=[1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
            top_k=2,
        )
        scores = result[0]
        assert scores.logprobs == [0.0, -4.0]
        ids = np.frombuffer(scores.topk_token_ids.data, dtype=np.int32).reshape(scores.topk_token_ids.shape)
        logprobs = np.frombuffer(scores.topk_logprobs.data, dtype=np.float32).reshape(scores.topk_logprobs.shape)
        assert ids.tolist() == [[0, 0], [9, 8]]
        np.testing.assert_allclose(logprobs[1], [-0.1, -0.2])
        assert 2 not in ids[1]
        assert fake_client.calls[0]["body"]["sampling_params"]["prompt_logprobs"] == 2
        assert fake_client.calls[0]["body"]["sampling_params"]["flat_logprobs"] is True
        assert fake_client.calls[0]["body"]["sampling_params"]["detokenize"] is False
        assert fake_client.calls[0]["body"]["prime_rl_compact_prompt_logprobs"] is True
        assert fake_client.closed is True

    asyncio.run(_run())


def _base64_tensor(array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": pybase64.b64encode(memoryview(array)).decode("ascii"),
    }


def test_compute_teacher_topk_decodes_compact_response(monkeypatch):
    async def _run():
        sampled = np.array([0.0, -4.0], dtype=np.float32)
        ids = np.array([[0, 0], [9, 8]], dtype=np.int32)
        logprobs = np.array([[-1e9, -1e9], [-0.1, -0.2]], dtype=np.float32)
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-compact",
                "choices": [],
                "prompt_logprobs": None,
                "prompt_logprobs_compact": {
                    "sampled_logprobs": _base64_tensor(sampled),
                    "topk_token_ids": _base64_tensor(ids),
                    "topk_logprobs": _base64_tensor(logprobs),
                },
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)
        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[False],
            completion_ids=[2],
            completion_mask=[True],
            completion_logprobs=[-0.5],
            completion_temperatures=[1.0],
            env_name="test-env",
        )

        scores = (
            await orchestrator_utils.compute_teacher_logprobs(
                clients=[vf.ClientConfig()],
                model_name="teacher-model",
                samples=[sample],
                top_k=2,
            )
        )[0]

        np.testing.assert_allclose(scores.logprobs, sampled)
        assert scores.topk_token_ids is not None
        assert scores.topk_logprobs is not None
        decoded_ids = np.frombuffer(scores.topk_token_ids.data, dtype=np.int32).reshape(scores.topk_token_ids.shape)
        decoded_logprobs = np.frombuffer(scores.topk_logprobs.data, dtype=np.float32).reshape(
            scores.topk_logprobs.shape
        )
        np.testing.assert_array_equal(decoded_ids, ids)
        np.testing.assert_allclose(decoded_logprobs, logprobs)
        assert fake_client.closed is True

    asyncio.run(_run())


def test_decode_compact_tensor_rejects_wrong_shape():
    payload = _base64_tensor(np.zeros((2, 2), dtype=np.float32))
    with np.testing.assert_raises_regex(ValueError, "shape"):
        orchestrator_utils._decode_compact_tensor(
            payload,
            name="topk_logprobs",
            expected_dtype="float32",
            expected_shape=[2, 3],
        )
