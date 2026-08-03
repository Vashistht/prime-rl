# Experiment status

Last updated: 2026-08-02 18:10 PDT.

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
  optimizer steps 0 and 1 completed without errors. At step 0, the corrected
  token-weighted teacher/student divergence was `0.18837`, sampled reverse KL
  was `0.18874`, and student entropy was `0.25959`. This independently matches
  the paper's reported external-teacher initial KL of approximately `0.19`.

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
- Safety walltime: 24 hours for RL and 26 hours for the teacher; all scientific
  settings remain matched and BF16-only.

## Prior control

The completed 25-step full-math/STEM run optimized teacher-top-64 union the
sampled token without the Eq. 5 correction. It is retained as an alternate
objective control, not labeled as the paper-faithful top-k result:

<https://wandb.ai/nvidia/opd_alignment-vashisth/runs/22e96c7cbe7a4d46a074f7b61fd648c6>
