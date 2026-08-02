import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.types import TrainRollout


def _train_rollout(policy_version: int) -> TrainRollout:
    return TrainRollout(
        raw={},
        env_name="test",
        example_id=0,
        group_id=uuid.uuid4(),
        policy_version=policy_version,
        off_policy_steps=0,
    )


def test_dispatch_gate_relaxes_only_for_terminal_penultimate_policy():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(max_steps=25, max_off_policy_steps=1)
    orchestrator.progress = SimpleNamespace(step=23)
    orchestrator.policy = SimpleNamespace(version=22)
    orchestrator.dispatcher = SimpleNamespace(dispatch_allowed=asyncio.Event())
    orchestrator.dispatcher.dispatch_allowed.set()

    # The usual one-step gate remains in force before the terminal batch.
    orchestrator.update_dispatch_gate()
    assert not orchestrator.dispatcher.dispatch_allowed.is_set()

    # No final v24 broadcast will arrive, so batch 24 may use policy v23.
    orchestrator.progress.step = 24
    orchestrator.policy.version = 23
    orchestrator.update_dispatch_gate()
    assert orchestrator.dispatcher.dispatch_allowed.is_set()

    # The terminal exception never admits a policy beyond the configured lag.
    orchestrator.dispatcher.dispatch_allowed.clear()
    orchestrator.policy.version = 22
    orchestrator.update_dispatch_gate()
    assert not orchestrator.dispatcher.dispatch_allowed.is_set()


def test_terminal_penultimate_policy_reaches_batch_24_then_drains():
    async def run() -> None:
        rollout = _train_rollout(policy_version=23)
        batch = object()
        out_q: asyncio.Queue = asyncio.Queue()
        out_q.put_nowait(rollout)

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(max_steps=25, max_off_policy_steps=1)
        orchestrator.progress = SimpleNamespace(step=24)
        orchestrator.policy = SimpleNamespace(version=23)
        orchestrator.stopped = asyncio.Event()
        orchestrator.draining = False
        orchestrator.train_sink = SimpleNamespace(add=AsyncMock(return_value=batch))
        orchestrator.dispatcher = SimpleNamespace(
            is_idle=True,
            out_q=out_q,
            dispatch_allowed=asyncio.Event(),
            disable_train_scheduling=Mock(),
            cancel_inflight_train_rollouts=AsyncMock(return_value=0),
        )

        async def finalize_train_batch(received_batch) -> None:
            assert received_batch is batch
            orchestrator.progress.step += 1

        orchestrator.finalize_train_batch = AsyncMock(side_effect=finalize_train_batch)

        orchestrator.update_dispatch_gate()
        assert orchestrator.dispatcher.dispatch_allowed.is_set()
        await orchestrator.main_loop()

        orchestrator.train_sink.add.assert_awaited_once_with(rollout, min_policy_version=23)
        orchestrator.finalize_train_batch.assert_awaited_once_with(batch)
        assert orchestrator.progress.step == 25
        assert orchestrator.draining is True
        assert orchestrator.stopped.is_set()
        orchestrator.dispatcher.disable_train_scheduling.assert_called_once_with()
        orchestrator.dispatcher.cancel_inflight_train_rollouts.assert_awaited_once_with()

    asyncio.run(run())


def test_main_loop_drains_at_max_steps_without_final_policy_broadcast():
    async def run() -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.config = SimpleNamespace(max_steps=1)
        orchestrator.progress = SimpleNamespace(step=1)
        orchestrator.stopped = asyncio.Event()
        orchestrator.draining = False
        orchestrator.dispatcher = SimpleNamespace(
            is_idle=True,
            disable_train_scheduling=Mock(),
            cancel_inflight_train_rollouts=AsyncMock(return_value=3),
        )

        await orchestrator.main_loop()

        assert orchestrator.draining is True
        assert orchestrator.stopped.is_set()
        orchestrator.dispatcher.disable_train_scheduling.assert_called_once_with()
        orchestrator.dispatcher.cancel_inflight_train_rollouts.assert_awaited_once_with()

    asyncio.run(run())


def test_dispatcher_is_not_idle_while_group_awaits_client():
    dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
    dispatcher.eval_source = []
    dispatcher.inflight = {}
    dispatcher.groups = {uuid.uuid4(): SimpleNamespace(kind="eval")}
    dispatcher.out_q = asyncio.Queue()

    assert dispatcher.is_idle is False


def test_drain_cancels_train_group_awaiting_client_selection():
    async def run() -> None:
        client_selection_started = asyncio.Event()
        release_client = asyncio.Event()

        class DelayedPool:
            async def select_train_client(self, _load):
                client_selection_started.set()
                await release_client.wait()
                return SimpleNamespace()

        dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
        dispatcher.training_mode = "opd"
        dispatcher.policy = SimpleNamespace(model_name="student")
        dispatcher.inference = DelayedPool()
        dispatcher.inflight = {}
        dispatcher.inflight_permits = 0
        dispatcher.train_scheduling_disabled = False
        dispatcher.metrics = SimpleNamespace(record_cancellation=Mock())

        group_id = uuid.uuid4()
        group = SimpleNamespace(kind="train", pinned_client=None)
        dispatcher.groups = {group_id: group}
        schedule_task = asyncio.create_task(dispatcher.schedule_group_rollout(group_id, group))
        await client_selection_started.wait()

        dispatcher.disable_train_scheduling()
        cancelled = await dispatcher.cancel_inflight_train_rollouts()
        release_client.set()

        assert await schedule_task is False
        assert cancelled == 0
        assert group_id not in dispatcher.groups
        assert dispatcher.inflight == {}

    asyncio.run(run())
