import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import TrainRollout


class _TrainEnvs:
    def __init__(self, group_size: int = 1) -> None:
        self.env = SimpleNamespace(
            config=SimpleNamespace(group_size=group_size),
            requires_group_scoring=False,
            advantage_fn=None,
            sampling_args={"temperature": 1.0},
        )

    def get(self, _env_name: str):
        return self.env


def _make_sink(*, batch_size: int | None = 3, token_batch_size: int | None = None) -> TrainSink:
    sink = TrainSink(
        SimpleNamespace(training_mode="opd", output_dir=Path("unused")),
        tokenizer=None,
        renderer=None,
        train_envs=_TrainEnvs(),
        mm_token_type_ids_mapping=None,
        batch_size=batch_size,
        token_batch_size=token_batch_size,
        pre_filters=[],
        post_filters=[],
    )
    sink.process_rollout = AsyncMock()
    return sink


def _rollout(policy_version: int, *, tokens: int = 2, group_id: uuid.UUID | None = None) -> TrainRollout:
    return TrainRollout(
        raw={
            "reward": 0.0,
            "trajectory": [],
            "token_usage": {"final_input_tokens": 1, "final_output_tokens": tokens - 1},
        },
        env_name="test",
        example_id=0,
        group_id=group_id or uuid.uuid4(),
        policy_version=policy_version,
        off_policy_steps=0,
    )


def test_stale_arrival_is_rejected_before_tokenization() -> None:
    sink = _make_sink()

    batch = asyncio.run(sink.add(_rollout(4), min_policy_version=6))

    assert batch is None
    assert sink.pending_batch == []
    assert sink.pending_groups == {}
    sink.process_rollout.assert_not_awaited()


def test_stale_buffer_is_purged_and_batch_refills_to_full_size() -> None:
    sink = _make_sink(batch_size=3)
    sink.pending_batch = [_rollout(4), _rollout(6)]

    assert asyncio.run(sink.add(_rollout(6), min_policy_version=6)) is None
    batch = asyncio.run(sink.add(_rollout(7), min_policy_version=6))

    assert batch is not None
    assert len(batch.rollouts) == 3
    assert [r.policy_version for r in batch.rollouts] == [6, 6, 7]


def test_stale_partial_group_is_purged_as_a_unit() -> None:
    sink = _make_sink(batch_size=2)
    stale_group_id = uuid.uuid4()
    sink.pending_groups[stale_group_id] = [_rollout(4, group_id=stale_group_id)]

    assert asyncio.run(sink.add(_rollout(6), min_policy_version=6)) is None

    assert stale_group_id not in sink.pending_groups
    assert [r.policy_version for r in sink.pending_batch] == [6]


def test_stale_tokens_do_not_satisfy_token_batch_threshold() -> None:
    sink = _make_sink(batch_size=None, token_batch_size=6)
    sink.pending_batch = [_rollout(4, tokens=100), _rollout(6, tokens=2)]

    assert asyncio.run(sink.add(_rollout(6, tokens=2), min_policy_version=6)) is None
    batch = asyncio.run(sink.add(_rollout(6, tokens=2), min_policy_version=6))

    assert batch is not None
    assert all(r.policy_version >= 6 for r in batch.rollouts)
    assert (
        sum(
            r.raw["token_usage"]["final_input_tokens"] + r.raw["token_usage"]["final_output_tokens"]
            for r in batch.rollouts
        )
        >= 6
    )
