# Experiment status

Last updated: 2026-08-02 21:46 PDT.

## Code

- Branch: `exp/mopd-eq5-dapo-qwen`
- Eq. 5 commit: `050e5bf33`
- Remote: `Vashistht/prime-rl`
- Focused pure-loss/config/sequence tests: 8 passed.
- Container preflight, including fused sparse value and gradient parity:
  17 passed.

## Original DAPO Eq. 5 run (operationally interrupted)

- Student: `Qwen/Qwen3-30B-A3B`
- Teacher: `Qwen/Qwen3-235B-A22B`
- Data: DAPO-Math-17k, 17,916 rows after one AIME26 exclusion
- Slurm teacher job: `2824700` (two nodes)
- Slurm Prime-RL job: `2825083` (12 nodes)
- W&B: <https://wandb.ai/nvidia/opd_alignment-vashisth/runs/a49312a48fa144d7b019f25e14d6d437>
- Checkpoint evaluator: watches steps 5/10/15/20/25 and submits serial
  AIME25/AIME26/GPQA-Diamond avg@8 jobs.
- The corrected token-weighted teacher/student divergence at step 0 was
  `0.18837`, sampled reverse KL was `0.18874`, and student entropy was
  `0.25959`. This independently matches the paper's reported external-teacher
  initial KL of approximately `0.19`.
- Step 5 checkpoint: stable at 18:27 PDT. Checkpoint eval Slurm `2825729`
  completed successfully with zero request/runtime errors: AIME25 `76.25%`
  (183/240), AIME26 `70.83%` (170/240), and GPQA-Diamond `62.50%`
  (990/1,584). The matching student baseline was `73.75%`, `74.58%`, and
  `62.94%`, respectively, so step 5 is mixed/approximately flat rather than
  collapsed.
- Step 5 training health: 2,048/2,048 sequences trainable, zero rollout
  errors, max policy lag 1. A small `1.03%` tail (21/2,048) reached the paper's
  32,768-token cap; mean total/decode lengths were 10,643/10,526, so responses
  are not piling up at the cap.
- Step 5 eval length audit: 19/240 AIME25 and 17/240 AIME26 samples reached
  the paper-matched 32,768-token cap; GPQA-Diamond had no truncations. The
  complete evaluator allocation averaged `90.3%` GPU utilization including
  startup/shutdown and `97.4%` over active samples.
- Step 10 checkpoint/eval: stable and completed successfully (Slurm
  `2825946`, exit 0). Avg@8 results were AIME25 `71.25%` (171/240), AIME26
  `75.00%` (180/240), and GPQA-Diamond `63.19%` (1,001/1,584). Baseline was
  `73.75%`, `74.58%`, and `62.94%`, so the trajectory remains mixed rather
  than collapsed. AIME25/AIME26 had 24/15 capped samples; GPQA had none.
  End-to-end evaluator GPU utilization was `90.0%`, or `96.9%` over active
  samples.
- At step 10 the Eq. 5 divergence was `0.18510`, sampled reverse KL was
  `0.18599`, and entropy was `0.26604`, versus `0.18837`, `0.18874`, and
  `0.25959` initially. These diagnostics do not yet show the paper's reported
  catastrophic regime.
- Step 15 checkpoint/eval: stable and completed successfully (Slurm
  `2826145`, exit 0). Avg@8 results were AIME25 `73.33%` (176/240), AIME26
  `74.17%` (178/240), and GPQA-Diamond `61.87%` (980/1,584). AIME25/AIME26
  had 19/14 capped samples and GPQA had one. At step 15, Eq. 5 divergence was
  `0.18248`, sampled reverse KL was `0.18294`, and entropy was `0.27317`.
- Training completed optimizer update 17 without numerical errors, rollout
  errors, or evidence of collapse. At approximately 21:17 PDT, Slurm
  preempted teacher job `2824700` and requeued it on different hosts. The
  running Prime-RL job retained the original fixed teacher endpoints and
  consequently failed with connection errors during update 18 (Slurm
  `2825083`, exit 143). This is an infrastructure interruption, not the
  scientific failure mode under test.
- This run saved HF weights only. Stable steps 5, 10, and 15 are valid, but no
  optimizer/scheduler state exists, so an exact continuation is impossible.

## Active clean DAPO recovery run

- Run name:
  `table4-qwen30b-original235b-mopd-eq5-topk64-dapo17k-b2048-g2-rerun1-20260802`.
- Teacher: Slurm `2826432`, one TP4 node. The compact live preflight passed
  with HTTP 200 and validated teacher-top-64 token/log-prob tensor shapes.
- Prime-RL: Slurm `2826440`, 12 nodes (10 generation, 2 trainer), submitted
  after the live teacher gate and running since 21:40 PDT.
- W&B:
  <https://wandb.ai/nvidia/opd_alignment-vashisth/runs/cab787732c794f52bf13126cf9691dfe>
- Scientific settings are unchanged from the original run: exact student and
  teacher snapshots, DAPO split, Eq. 5 top-64 objective, batch 2,048 as 1,024
  prompts x 2 samples, 32,768-token cap, temperature 1.0 without sampling
  truncation, LR `1e-6`, 25 updates, and BF16 only.
- The recovery uses one teacher node because the two original teacher nodes
  averaged only about 33% wall-clock GPU utilization each. One node reduces
  allocation/preemption exposure while retaining the same TP4 server and
  projects to roughly 67% combined wall-clock utilization.
- Trainer checkpoints now include optimizer, scheduler, progress, and
  dataloader state every five steps (`weights_only = false`) in addition to HF
  weight exports. The persistent evaluator watches stable steps 5/10/15/20/25
  and submits serialized AIME25/AIME26/GPQA-Diamond avg@8 jobs.
- Startup completed cleanly: all 40/40 BF16, unquantized student inference
  replicas became ready, the full-state trainer initialized from step 0, and
  the first 2,048-sample rollout entered steady state with all 40 generation
  GPUs at approximately 99% utilization. Trainer and teacher GPUs are idle in
  this generation phase by design; they become active during the subsequent
  teacher-scoring and optimizer-update phases.

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
- Prime-RL: Slurm `2825644`, completed successfully through step 25 in 43:09;
  all five requested checkpoints are stable.
- W&B:
  <https://wandb.ai/nvidia/opd_alignment-vashisth/runs/3c07cdf8cc0b4081b5fdcce5b3dca07e>
- Checkpoint evaluator: persistent watcher for steps 5/10/15/20/25, all three
  benchmarks avg@8.
- All checkpoint evals completed successfully with zero runtime errors:

  | Checkpoint | AIME25 | AIME26 | GPQA-Diamond |
  | --- | ---: | ---: | ---: |
  | Baseline | 4.58% | 6.67% | 19.95% |
  | Step 5 | 4.17% | 6.67% | 19.19% |
  | Step 10 | 3.75% | 4.17% | 18.62% |
  | Step 15 | 3.33% | 4.58% | 18.69% |
  | Step 20 | 5.00% | 3.33% | 19.32% |
  | Step 25 | 2.92% | 5.00% | 18.94% |

  The Base control degrades overall but does not exhibit a numerical crash.
- The Base teacher service `2825629` was released immediately after training;
  checkpoint evaluation uses only the saved student weights.
- W&B currently labels this Base run `crashed`, but Slurm `2825644` exited 0,
  checkpoint 25 is stable, and all checkpoint evaluations completed. Treat the
  dashboard state as a multi-process finalization artifact, not a run failure.
- Safety walltime: 24 hours for RL and 26 hours for the teacher; all scientific
  settings remain matched and BF16-only.

## Prior control

The completed 25-step full-math/STEM run optimized teacher-top-64 union the
sampled token without the Eq. 5 correction. It is retained as an alternate
objective control, not labeled as the paper-faithful top-k result:

<https://wandb.ai/nvidia/opd_alignment-vashisth/runs/22e96c7cbe7a4d46a074f7b61fd648c6>
