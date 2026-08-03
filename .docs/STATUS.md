# Experiment status

Last updated: 2026-08-02 18:40 PDT.

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
- Current state at this update: both teacher and Prime-RL jobs are running;
  optimizer updates through policy version 6 completed without errors. At
  step 0, the corrected token-weighted teacher/student divergence was
  `0.18837`, sampled reverse KL was `0.18874`, and student entropy was
  `0.25959`. This independently matches the paper's reported external-teacher
  initial KL of approximately `0.19`.
- Step 5 checkpoint: stable at 18:27 PDT; checkpoint eval Slurm `2825729` is
  running. Its completed AIME25 avg@8 result is `76.25%` (183/240) with zero
  errors; AIME26 and GPQA-Diamond remain in progress.
- Step 5 training health: 2,048/2,048 sequences trainable, zero rollout
  errors, max policy lag 1. A small `1.03%` tail (21/2,048) reached the paper's
  32,768-token cap; mean total/decode lengths were 10,643/10,526, so responses
  are not piling up at the cap.

## Matched Base-student control

Requested comparison:

- student: `Qwen/Qwen3-30B-A3B-Base`;
- teacher, Eq. 5 objective, hyperparameters, topology, and evaluation protocol:
  identical to the active run;
- data: the older complete 16,818-row local math/STEM view.

- Exact Base snapshot:
  `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` (16/16 shards and all 18,867
  indexed tensors verified).
- Base baseline eval: Slurm `2825618`, completed with zero errors. Avg@8
  results: AIME25 `4.58%`, AIME26 `6.67%`, GPQA-Diamond `19.95%`.
- Replacement teacher service: Slurm `2825629`, two TP4 nodes, live compact
  top-64 probes passed. The initial `2825619` allocation was intentionally
  cancelled before RL submission solely to extend walltime.
- Prime-RL: Slurm `2825644`, running on 12 nodes.
- W&B:
  <https://wandb.ai/nvidia/opd_alignment-vashisth/runs/3c07cdf8cc0b4081b5fdcce5b3dca07e>
- Checkpoint evaluator: persistent watcher for steps 5/10/15/20/25, all three
  benchmarks avg@8.
- Stable checkpoints at this update: steps 5, 10, and 15. Step 5 eval Slurm
  `2825705` completed successfully; step 10 eval `2825730` is running and step
  15 eval `2825738` is serialized behind it.
- Step 5 avg@8 results (zero errors): AIME25 `4.17%`, AIME26 `6.67%`,
  GPQA-Diamond `19.19%`. Relative to the baseline above, there is no early
  improvement; these small deltas remain within avg@8 sampling noise.
- Safety walltime: 24 hours for RL and 26 hours for the teacher; all scientific
  settings remain matched and BF16-only.

## Prior control

The completed 25-step full-math/STEM run optimized teacher-top-64 union the
sampled token without the Eq. 5 correction. It is retained as an alternate
objective control, not labeled as the paper-faithful top-k result:

<https://wandb.ai/nvidia/opd_alignment-vashisth/runs/22e96c7cbe7a4d46a074f7b61fd648c6>
