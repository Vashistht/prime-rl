# Experiment status

Last updated: 2026-08-02 17:14 PDT.

## Code

- Branch: `exp/mopd-eq5-dapo-qwen`
- Eq. 5 commit: `050e5bf33`
- Remote: `Vashistht/prime-rl`
- Focused pure-loss/config/sequence tests: 8 passed.
- Container preflight, including fused sparse value and gradient parity:
  17 passed.

## Active DAPO Eq. 5 run

- Student: `Qwen/Qwen3-30B-A3B`
- Teacher: `Qwen/Qwen3-235B-A22B`
- Data: DAPO-Math-17k, 17,916 rows after one AIME26 exclusion
- Slurm teacher job: `2824700` (two nodes)
- Slurm Prime-RL job: `2825083` (12 nodes)
- W&B: <https://wandb.ai/nvidia/opd_alignment-vashisth/runs/a49312a48fa144d7b019f25e14d6d437>
- Checkpoint evaluator: watches steps 5/10/15/20/25 and submits serial
  AIME25/AIME26/GPQA-Diamond avg@8 jobs.
- Current state at this update: both teacher and Prime-RL jobs running; all
  student and teacher endpoints ready; rollout batch zero generating.

## Matched Base-student control

Requested comparison:

- student: `Qwen/Qwen3-30B-A3B-Base`;
- teacher, Eq. 5 objective, hyperparameters, topology, and evaluation protocol:
  identical to the active run;
- data: the older complete 16,818-row local math/STEM view.

The Base snapshot is being staged and verified. This control has not yet been
submitted; it must not be pointed at the post-trained 30B snapshot as a
fallback.

## Prior control

The completed 25-step full-math/STEM run optimized teacher-top-64 union the
sampled token without the Eq. 5 correction. It is retained as an alternate
objective control, not labeled as the paper-faithful top-k result:

<https://wandb.ai/nvidia/opd_alignment-vashisth/runs/22e96c7cbe7a4d46a074f7b61fd648c6>
