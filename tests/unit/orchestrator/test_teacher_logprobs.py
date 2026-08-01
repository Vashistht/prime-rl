import asyncio
import json

import httpx
import numpy as np
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

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=200,
            content=json.dumps(self._payload).encode(),
            request=request,
        )


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
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]

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
                        # realized token.
                        "9": {"logprob": -0.1},
                        "2": {"logprob": -4.0},
                        "8": {"logprob": -0.2},
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

    asyncio.run(_run())
