import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "src/prime_rl/templates"


def render_multi_node_rl(*, orchestrator_on_inference: bool, cleanup_grace_period: int = 0) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template("multi_node_rl.sbatch.j2")
    return template.render(
        account="test-account",
        backend_port=8100,
        config_dir="/tmp/config",
        cleanup_grace_period=cleanup_grace_period,
        decode_env_overrides={},
        decode_port=8200,
        decode_vllm_extra_json="",
        exclude=None,
        gpus_per_node=4,
        inference_data_parallel_rpc_port=29600,
        inference_enable_expert_parallel=True,
        inference_tp=1,
        is_disaggregated=False,
        job_name="lifecycle-test",
        kv_offload=False,
        kv_offload_mooncake=False,
        nodelist=None,
        nodes_per_infer_replica=2,
        num_decode_nodes=0,
        num_decode_replicas=0,
        num_infer_nodes=4,
        num_infer_replicas=2,
        num_prefill_nodes=0,
        num_prefill_replicas=0,
        num_train_nodes=2,
        orchestrator_on_inference=orchestrator_on_inference,
        orchestrator_output_dir="/tmp/out/run",
        output_dir="/tmp/out",
        partition="batch",
        pre_run_command=None,
        prefill_env_overrides={},
        prefill_port=8000,
        prefill_vllm_extra_json="",
        project_dir="/tmp/prime-rl",
        ranks_filter="0",
        router=SimpleNamespace(
            port=8000,
            type="vllm",
            policy="round_robin",
            decode_sidecar_port=9000,
        ),
        time="00:10:00",
        use_deep_gemm=False,
        use_nccl_broadcast=True,
    )


@pytest.mark.parametrize("orchestrator_on_inference", [False, True])
def test_multi_node_rl_lifecycle_is_valid_bash(orchestrator_on_inference: bool):
    script = render_multi_node_rl(orchestrator_on_inference=orchestrator_on_inference)

    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("orchestrator_on_inference", "expected_assignment"),
    [
        (False, "ORCH_PROCID=$NUM_INFER_NODES"),
        (True, "ORCH_PROCID=$((NUM_INFER_NODES - 1))"),
    ],
)
def test_multi_node_rl_clean_shutdown_waits_for_every_trainer(
    orchestrator_on_inference: bool,
    expected_assignment: str,
):
    script = render_multi_node_rl(orchestrator_on_inference=orchestrator_on_inference)

    assert expected_assignment in script
    assert "wait -n -p FINISHED_PID" in script
    assert 'touch "$LIFECYCLE_DIR/trainer_${TRAIN_NODE_RANK}.done"' in script
    assert "for ((rank=0; rank<NUM_TRAIN_NODES; rank++)); do" in script
    assert 'touch "$LIFECYCLE_DIR/shutdown"' in script
    assert "job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}" in script
    assert 'if [ "$IS_INFER_NODE" -eq 1 ]; then' in script
    assert "sleep 3600" not in script


def test_multi_node_rl_failure_preserves_configured_cleanup_grace_period():
    script = render_multi_node_rl(orchestrator_on_inference=False, cleanup_grace_period=37)

    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert 'if [ "$raw_code" -ne 0 ]; then' in script
    assert "sleep 37" in script
    assert script.index("sleep 37") < script.index('terminate_local_components\n    exit "$code"')


def test_multi_node_rl_zero_cleanup_grace_has_no_failure_sleep():
    script = render_multi_node_rl(orchestrator_on_inference=False, cleanup_grace_period=0)

    assert 'if [ "$raw_code" -ne 0 ]; then' not in script
    assert "checkpoints to flush" not in script
