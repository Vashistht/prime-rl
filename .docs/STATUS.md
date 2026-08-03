# Experiment status

Last updated: 2026-08-03 10:10 PDT.

## Code

- Branch: `exp/mopd-eq5-dapo-qwen`
- Eq. 5 commit: `050e5bf33`
- Single-environment data-position resume fix: `9282e69e3`
- Remote: `Vashistht/prime-rl`
- Focused pure-loss/config/sequence tests: 8 passed.
- Container preflight, including fused sparse value and gradient parity:
  17 passed.
- Data-position replay tests: 4 passed, including the exact DAPO
  17,916-row/10,240-prompt checkpoint boundary. The wider host-side
  orchestrator suite cannot collect on the login node because its installed
  FlashAttention CUDA extension cannot be mapped there; formatting, lint, and
  Python compilation pass.

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

## DAPO full-state recovery run and step-10 continuation

- Run name:
  `table4-qwen30b-original235b-mopd-eq5-topk64-dapo17k-b2048-g2-rerun1-20260802`.
- This is a fresh run after the original job's teacher-service preemption. The
  original interruption was infrastructural, not a scientific collapse; this
  recovery is configured with resumable full-state checkpoints.
- Initial teacher: Slurm `2826432`, one TP4 node. The compact live preflight passed
  with HTTP 200 and validated teacher-top-64 token/log-prob tensor shapes.
- Initial Prime-RL segment: Slurm `2826440`, 12 nodes (10 generation, 2
  trainer), submitted after the live teacher gate. W&B:
  <https://wandb.ai/nvidia/opd_alignment-vashisth/runs/cab787732c794f52bf13126cf9691dfe>
- Scientific settings are unchanged from the original run: student
  `Qwen/Qwen3-30B-A3B` at
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, teacher
  `Qwen/Qwen3-235B-A22B` at
  `8efa61729e24bd65b1d152b5ab5409052aa80e65`, 17,916-row leak-filtered
  DAPO-Math-17k split, Eq. 5 teacher-top-64 objective, batch 2,048 as 1,024
  prompts x 2 samples, 32,768-token cap, temperature 1.0 without sampling
  truncation, LR `1e-6`, 25 updates, and BF16 only.
- This segment used one teacher node to reduce the allocation, but the measured
  steady-state audit showed that it became the critical path. More importantly,
  the `normal` QOS protection expires after 4:05, making a one-teacher
  continuation predictably too long for the protected window.
- Steps 0 through 4 completed cleanly with 2,048/2,048 trainable sequences per
  update, zero rollout errors, and maximum policy lag 1. During their active
  phases, all 40 generation replicas ran at approximately 99% GPU utilization,
  trainer throughput was approximately 45--46.6K token/s, and the teacher ran
  at approximately 95--99% GPU utilization.
- At step 4, corrected Eq. 5 divergence was `0.1874304414`, sampled reverse KL
  was `0.1882639974`, and student entropy was `0.2615351081`; training remains
  numerically healthy.
- The step 5 full-state checkpoint is durable: its DCP payload is 171 GiB and
  indexes 2,340 keys spanning model, optimizer, scheduler, and progress state.
  The matching HF export has a `STABLE` marker. This confirms that the fresh
  model, optimizer, scheduler, and progress state can be restored, unlike the
  interrupted weights-only run.
- Trainer update 5 also completed cleanly: 2,048/2,048 trainable sequences,
  zero errors, maximum policy lag 1, `0.8%` truncation, entropy `0.2668`, and
  approximately 46.1K token/s (`14.6%` reported MFU).
- The persistent evaluator watches stable steps 5/10/15/20/25 and submits
  serialized AIME25/AIME26/GPQA-Diamond avg@8 jobs. Step 5 evaluation Slurm
  `2827011` completed with exit 0 and zero runtime errors: AIME25 `72.50%`
  (174/240), AIME26 `77.08%` (185/240), and GPQA-Diamond `63.89%`
  (1,012/1,584). Relative to the matched baseline (`73.75%`, `74.58%`, and
  `62.94%`), the checkpoint is mixed/approximately flat rather than collapsed.
- Step 10 is also fully durable: the 171 GiB DCP contains the same 2,340-key
  model/optimizer/scheduler/progress structure as step 5, and its 13-shard HF
  export is marked `STABLE`. Its saved progress is 20,480 samples from 10,240
  source prompts and 213,570,652 tokens.
- Step 10 evaluation Slurm `2827719` completed with exit 0 and zero runtime
  errors: AIME25 `71.25%` (171/240), AIME26 `75.00%` (180/240), and
  GPQA-Diamond `63.26%` (1,002/1,584). These remain approximately flat against
  both baseline and step 5; there is no failure-mode collapse through step 10.
- From the first update to checkpoint 10, corrected Eq. 5 divergence moved
  from `0.188573` to `0.182438`, sampled reverse KL from `0.188920` to
  `0.183279`, and entropy from `0.260466` to `0.265652`. The sampled token was
  inside the teacher top 64 for `99.9866%` of tokens at checkpoint 10.
- A five-second-sample utilization audit across 11 complete steady-state
  intervals measured `56.32%` GPU-time-weighted utilization across all 52
  GPUs. Student generation, trainer, and teacher roles averaged `59.26%`,
  `43.61%`, and `52.26%` wall-clock utilization, respectively, but each ran at
  `96.7--98.5%` when active and no individual node lagged its peers. The
  remaining idle time is synchronized pipeline waiting: the single TP4 235B
  teacher is now the critical path for about 35% of steady-state wall time.
  Adding more student-generation nodes alone would not improve throughput.
- At 01:38:47 PDT, teacher `2826432` was preempted immediately after its QOS
  exemption expired. The fixed endpoint disappeared, and Prime-RL `2826440`
  failed with an API connection error at 01:41:15. Trainer updates 10--12 had
  run, but only step 10 was a complete durable trainer/orchestrator/HF
  checkpoint; the partial future trajectory is intentionally discarded. This
  was an infrastructure failure, not model collapse.
- Prime-RL's legacy checkpoint restored aggregate progress but reset
  `TrainSource` to row zero. Commit `9282e69e3` fixes the single-environment
  case by deterministically advancing the source by the checkpoint's 10,240
  shipped prompts before dispatch. It refuses ambiguous multi-environment
  resumes. This preserves first-epoch DAPO coverage instead of repeating the
  first 10,240 prompts.
- Stale step-10--13 rollout files, NCCL rendezvous markers, and the forced
  step-13 orchestrator checkpoint were moved to
  `recovery_archive/pre_resume_step10_20260803`; the complete step-10 trainer,
  orchestrator, and HF checkpoints were preserved and their metadata hashes
  reverified after rendering.
- Continuation teacher Slurm `2828255` allocated two independent TP4 replicas;
  both loaded the exact teacher and passed the compact top-64 live probe. It
  then submitted the fresh 12-node Prime-RL continuation as Slurm `2828367`
  with `resume_step=10`, absolute `max_steps=25`, a new W&B identity, and
  otherwise identical scientific settings. Both jobs had `--no-requeue`.
- The scheduler initially forecast that `2828367` would wait until after the
  already-running teachers' 4:05 preemption protection expired. Rather than
  hold eight idle GPUs and knowingly repeat the infrastructure failure,
  teacher `2828255` was intentionally cancelled after validation and its logs
  and GPU traces were archived under
  `recovery_archive/retimed_teacher_2828255`.
- Subsequent replacement teachers were submitted with delayed begin times and
  kept dependency-gated 15 minutes ahead of `2828367`. Scheduler forecasts
  moved repeatedly as the provider drained the cluster. Every superseded
  teacher was cancelled while still pending and therefore consumed zero GPU
  time. No replacement teacher reached host publication, so the archived
  endpoints were never rebound and the RL continuation never started.
- At 09:43:32 PDT, the provider activated reservation
  `cspu_cluster_return`, putting all 324 DFW GPU nodes into maintenance with
  `IGNORE_JOBS`. At 10:03:20 Slurm system-cancelled the final pending teacher
  `2831208` and Prime-RL `2828367` as `CANCELLED by 0`; both had zero runtime
  and zero allocation. This was cluster retirement, not a model/training
  failure. Step 10 remains the latest durable recovery point and steps 15/20/25
  were not created.
- The DAPO evaluator completed only steps 5 and 10 (`2827011`, `2827719`),
  both with zero errors. Its persistent watcher was stopped after the parent
  job became terminal so it would not restart-loop on the retired cluster.
- DFW has no alternate GPU partition or account/QOS bypass. The documented
  migration target is `aws-cmh-slurm-1` via `aws-cmh-login-01` or
  `aws-cmh-login-02`. CMH is four-GPU-node ARM GB300/sm103, so the existing
  CMH-native Prime image must be revalidated with a one-step smoke before the
  step-10 DCP is resumed; the DFW GB200 image is not assumed ABI-portable.

## Queued matched post-trained old-data control

- This fills the missing comparison cell: the same post-trained
  `Qwen3-30B-A3B` student, original 235B teacher, corrected Eq. 5 objective,
  and DAPO-run settings, changing only the training data to the complete local
  math/STEM view used by the completed Base control.
- The processed dataset has exactly 16,818 rows and SHA-256
  `760b842c8d77127733534801fca43882959ff325fc85d895e507832520f6a95a`.
  Its manifest reports zero canonical containment matches against AIME25,
  AIME26, and GPQA-Diamond.
- A dry render verified BF16, no FP8 or transfer quantization, corrected
  `mopd_eq5`, teacher top-64, batch 2,048 as 1,024 prompts x 2 samples, LR
  `1e-6`, 32,768 tokens, 25 updates, and resumable full-state checkpoints at
  steps 5/10/15/20/25.
- CPU-only launch coordinator Slurm `2827170` had the verified dependency
  `afterany:2828367`. The system cancellation released that dependency at
  10:03:28. It rendered the matched configuration and submitted teacher
  `2831552`, which Slurm immediately system-cancelled before allocation because
  the nodes were reserved for maintenance. The launcher detected that the
  teacher ended before readiness and deliberately withheld the fresh RL
  submission. The coordinator exited after 53 seconds; no GPU was consumed
  and no partial control training occurred.
- Future generated teacher and Prime-RL jobs use Slurm `--no-requeue`: if
  preempted, they now fail visibly instead of silently restarting against
  stale fixed endpoints or replaying a checkpoint segment.
- The control evaluator watcher was stopped after the cluster-return incident.
  Recreate it on the migration cluster after a fresh RL job id exists.

## Matched Base-student control

Requested comparison:

- student: `Qwen/Qwen3-30B-A3B-Base`;
- teacher, Eq. 5 objective, scientific hyperparameters, and evaluation protocol:
  identical to the active run;
- data: the older complete 16,818-row local math/STEM view.

The Base allocation used two identical TP4 teacher-serving replicas rather
than one and saved weights-only checkpoints rather than full optimizer state.
Those are throughput/recovery differences only; they do not alter teacher
scores, the loss, gradients, or checkpoint evaluation.

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
