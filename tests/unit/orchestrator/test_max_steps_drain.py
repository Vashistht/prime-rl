import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.orchestrator import Orchestrator


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
