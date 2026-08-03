from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.train_source import TrainSource


class _FakeEnv:
    def __init__(
        self,
        name: str,
        example_ids: list[int],
        *,
        ratio: float,
        group_size: int = 1,
        requires_group_scoring: bool = False,
    ) -> None:
        self.name = name
        self._rows = [{"example_id": example_id} for example_id in example_ids]
        self.config = SimpleNamespace(ratio=ratio, group_size=group_size)
        self.requires_group_scoring = requires_group_scoring

    def get_dataset(self, seed: int | None = None) -> list[dict]:
        return list(self._rows)


def _key(example: dict) -> tuple[str, int]:
    return example["env_name"], example["example_id"]


def test_fast_forward_restores_uninterrupted_source_position() -> None:
    envs = [_FakeEnv("math", list(range(7)), ratio=1.0, group_size=2, requires_group_scoring=True)]
    uninterrupted = TrainSource(envs, seed=42)  # type: ignore[arg-type]
    expected = [_key(uninterrupted.next_example(2)) for _ in range(31)]  # type: ignore[arg-type]

    resumed = TrainSource(envs, seed=42)  # type: ignore[arg-type]
    resumed.fast_forward(19)
    actual = [_key(resumed.next_example(2)) for _ in range(12)]  # type: ignore[arg-type]

    assert actual == expected[19:]


def test_fast_forward_rejects_negative_count() -> None:
    source = TrainSource([_FakeEnv("math", [0], ratio=1.0)], seed=42)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative"):
        source.fast_forward(-1)


def test_fast_forward_rejects_ambiguous_multi_env_progress() -> None:
    source = TrainSource(
        [_FakeEnv("math", [0], ratio=0.5), _FakeEnv("stem", [1], ratio=0.5)],
        seed=42,
    )  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="multi-environment"):
        source.fast_forward(1)


def test_fast_forward_matches_dapo_step_10_position() -> None:
    envs = [_FakeEnv("math", list(range(17_916)), ratio=1.0)]
    uninterrupted = TrainSource(envs, seed=42)  # type: ignore[arg-type]
    for _ in range(10_240):
        uninterrupted.next_example(1)

    resumed = TrainSource(envs, seed=42)  # type: ignore[arg-type]
    resumed.fast_forward(10_240)

    assert resumed.cursors == {"math": 10_240}
    assert resumed.rng.getstate() == uninterrupted.rng.getstate()
    assert resumed.next_example(1) == uninterrupted.next_example(1)
