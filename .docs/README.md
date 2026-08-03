# MOPD Eq. 5 reproduction handoff

This branch adds the bias-corrected teacher-top-k objective from MOPD Eq. 5
to Prime-RL. It is intended for the external-teacher failure experiment with
`Qwen/Qwen3-30B-A3B` as student and `Qwen/Qwen3-235B-A22B` as teacher.

## What changed

Set the sparse objective explicitly:

```toml
[trainer.loss]
type = "default"
opd_top_k_objective = "mopd_eq5"

[orchestrator]
training_mode = "opd"
opd_top_k = 64
```

For teacher support `T_k`, the optimized token loss is

```text
sum(v in T_k) [p_student(v) log(p_student(v) / p_teacher(v))
               - p_student(v) + p_teacher(v)]
```

The gathered student and teacher probabilities remain normalized over the
full vocabulary. The realized student token is not added to the optimized
support. Prime-RL first averages over tokens in each completion and then over
completions, matching the sequence-balanced form in the paper.

The prior `topk_plus_sampled` behavior remains the default for compatibility.
It is an ablation matching Peter's teacher-top-k-union-sampled implementation,
not the paper's Eq. 5 arm.

The implementation also logs:

- `mopd_eq5_topk_divergence`: corrected Eq. 5 value, token-weighted.
- `mopd_eq5_sequence_mean`: the per-completion token mean; its W&B mean is
  the closest diagnostic to the optimized sequence-balanced objective.
- `topk_reverse_kl`: naive truncated reverse KL, diagnostic only.
- `sampled_reverse_kl`: Monte Carlo full-vocabulary reverse-KL proxy.
- student/teacher top-k mass and sampled-token top-k membership.

## New-cluster setup

Use a Slurm cluster with four NVIDIA GPUs per node and Pyxis/Enroot support.
The validated run used GB200 nodes. Keep all arithmetic in BF16: do not enable
FP8 in the trainer, inference engine, weight transfer, or DeepGEMM.

The container used for the validated run was named
`opd-primerl-gb200-evals.sqsh`. A replacement container must contain:

- the Prime-RL environment at `/opt/prime-rl-venv`;
- PyTorch and CUDA compatible with the target nodes;
- vLLM with prompt-logprob/top-k support (validated with vLLM 0.22.0);
- Prime-RL's editable packages, renderer package, passthrough environment,
  AIME2025, AIME2026, and GPQA environments;
- `verifiers`/`vf-eval`, FlashAttention, and FlashInfer.

Mount the repository, model cache, output directory, and dataset into the
container. Keep credentials outside Git. The launch environment expects
`WANDB_API_KEY`; `HF_TOKEN` is only needed when snapshots are not already
staged. The known-good runtime variables are conceptually:

```bash
export PRIMERL_CONTAINER=/path/to/opd-primerl-gb200-evals.sqsh
export PRIMERL_VENV=/opt/prime-rl-venv
export OPD_MOUNTS=/shared:/shared
export OPD_ENV_FILE=/secure/path/to/runtime.env
export WANDB_PROJECT=opd_alignment-vashisth
export WANDB_ENTITY=nvidia
```

Stage and fully verify these exact model repositories before requesting GPUs:

- student: `Qwen/Qwen3-30B-A3B`;
- external teacher: `Qwen/Qwen3-235B-A22B` (not a Thinking or Instruct
  suffixed substitute);
- optional control student: `Qwen/Qwen3-30B-A3B-Base`.

For every snapshot, require `config.json`,
`model.safetensors.index.json`, and every shard named by the index. The Qwen3
renderer is explicit and deterministic, so it does not depend on a tokenizer
chat-template string; it does require the Qwen3 ChatML special tokens.

## Experiment settings

The current DAPO arm uses:

- DAPO-Math-17k processed prompts, preserving order and native duplicates;
- 17,916 rows after excluding one exact AIME26 duplicate;
- batch size 2,048 = 1,024 source prompts x two independent rollouts;
- 25 optimizer steps, LR `1e-6`, max sequence/completion length 32,768;
- temperature 1.0, no top-p or top-k sampling truncation;
- 10 inference nodes (40 TP1 replicas), two trainer nodes (DP replicate 2,
  EP4), and two TP4 teacher nodes;
- checkpoint weights at steps 5, 10, 15, 20, and 25;
- AIME25, AIME26, and GPQA-Diamond checkpoint evaluation at avg@8.

This local dataset reaches its epoch boundary after about 17.5 steps at this
batch size, so steps 18-25 contain reshuffled repeated prompts. Record that
when interpreting any instability near the paper's reported step-18 collapse.

Before launching on another cluster, run the focused preflight inside the
target container:

```bash
python -m pytest -q \
  tests/unit/train/rl/test_topk_opd_loss.py \
  tests/unit/train/rl/test_fused_lm_head.py::test_fused_teacher_topk_mopd_eq5_matches_dense_forward_and_backward_cpu
```

The operational configs, launch wrappers, curated JSONL, and eval watcher are
currently in the cluster's `prime_rl_scripts` workspace. Move them into the
private `projects_opd` experiment repository once its remote is created; they
are intentionally not part of Prime-RL itself. See [STATUS.md](STATUS.md) for
the current run identifiers.
