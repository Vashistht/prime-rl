# SFT vs OPD vs RL: theory-first toy experiments

This directory is a small, CPU-friendly, first-principles study.  It is
intentionally not a new Prime-RL environment.  The organizing object is an
explicit desired/evaluation distribution `p_star`, kept separate from the
teacher `q_teacher` and learned student `p_theta`.  In the core experiments,
demonstrations are unfiltered teacher samples, so `p_demo = q_teacher`; this
removes a demo-versus-teacher confound and makes teacher bias relative to
`p_star` visible.  The experiments start from the actual update operators, use
the simplest distribution that can identify a mechanism, and let discrepancies
between prediction and observation determine the next experiment.

The supplied PDF contributed useful taxonomy and operator identities--support,
occupancy, KL direction, and teacher/reward mismatch--but not the experiment
template.  The benchmark families, controls, and branch rules here were
derived from first principles.  `prime_loss_checks.py` calls Prime-RL's real
loss functions so that the names in the notebook match the implementation.

The organizing hypothesis is conditional, not a universal training recipe:

- **SFT learns the demonstration law.** In the core setup that law is
  `p_demo = q_teacher`; a demonstration can expose useful actions and all of
  their prefixes without asking the student to discover them.
- **OPD learns from the teacher on student occupancy.** It is useful when
  student rollouts reach valuable error/recovery states absent from teacher
  demonstrations and the teacher remains reliable there.  In a one-state,
  realizable population problem, both KL directions still target `q_teacher`.
- **RL follows utility, not teacher mass.** It can move toward `p_star` when
  reward is appropriately calibrated, but useful outcomes must be reachable
  and reward must be reliable.  An unregularized `log p_star` reward is
  mode-seeking rather than distribution-matching.
- **Retention is method-agnostic at first order.** Forgetting depends on the
  realized update direction and size, replay/reference regularization, and
  parameter sharing; RL is not intrinsically protected.

## Files

- `DESIGN.md`: mathematical objects, staged hypotheses, fairness rules, and
  branch/stop criteria.
- `opd_toy_study.ipynb`: main narrative, predictions, experiments, plots, and
  conclusions (generated only after the staged controls pass).
- `opd_toy_study.py`: readable percent-cell source used to generate the
  notebook.
- `toy_methods.py`: the single source of truth for toy SFT, OPD, and RL
  operators, sampling, KL-matched steps, and shared metrics.
- `families.py`: transparent distribution and student-family definitions.
- `truth_mixture_experiments.py`: primary explicit-truth benchmark with
  geometrically decaying named-mode probabilities.  The trainable core is one
  categorical logit per reachable mode; the teacher can distort/floor modal
  weights, and Gaussian kernels only render the distributions.  Exact
  population objectives compare SFT, both PD KL directions, pure truth-reward
  RL, and entropy-calibrated truth RL before sampling noise.  An optional
  labeled joint model `p(mode, coordinate)` adds a single shared Gaussian
  translation parameter.  Exact categorical teacher zeros make reverse KL
  infinite unless the corresponding student mode is structurally excluded;
  the main notebook uses a displayed positive epsilon floor.
- `truth_sampling_experiments.py`: finite categorical-mode companion comparing
  teacher-sampled SFT, full-logit forward-PD, student-sampled reverse-PD, and
  sampled truth-reward RL against their exact initial gradients.  It reports
  gradient noise/SNR and verifies per-mode discovery
  `1 - (1 - probability)^batch_size` over repeated seeds.
- `truth_reward_experiments.py`: reward-semantics control comparing the
  existing `log p_star` utility with binary reference matching and squared
  distance to a truth-sampled reference.  It predicts each reward's pure and
  entropy-regularized endpoints, then checks Prime-style unnormalized
  group-relative advantages at a fixed completion budget.  All policy losses,
  sampling, and group centering still come from `toy_methods.py`.
- `mixture_experiments.py`: capacity-limited mass/coverage/peak geometry.
- `shift_support_experiments.py`: graded tails versus overlap-only discovery.
- `two_step_experiments.py`: factorial state collector, KL direction, and
  teacher-reliability controls.
- `target_curriculum_experiments.py`: teacher/utility mismatch and an analytic
  distillation-to-RL handoff at fixed calls versus matched movement.
- `retention_experiments.py`: exact feature-overlap forgetting null control.
- `topk_experiments.py`: implementation-specific sparse-OPD audit.
- Every experiment module calls `toy_methods`; none reimplements the basic
  SFT, OPD, or RL training losses.
- `prime_loss_checks.py`: CPU checks against the real Prime-RL SFT, OPD, and
  RL losses, including the current teacher-top-k-plus-sampled OPD variant.

The shortest code-reading path is `toy_methods.py` for the four shared losses,
then `truth_mixture_experiments.py::population_loss` for the small dispatch
that applies them to `p_star`, `q_teacher`, and `p_theta`.  The remaining code
is explicit validation, experiment records, uncertainty summaries, and plots;
it does not hide another trainer or environment.

## Reproduce

From this directory, use the existing Prime-RL runtime (it already contains
PyTorch) and add only notebook/plotting packages through `uv`:

```bash
RUNTIME=../../.venv/bin/python
MALLOC_ARENA_MAX=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
UV_CACHE_DIR=/tmp/opd-toy-uv-cache uv run --no-project --python "$RUNTIME" \
  --with 'matplotlib>=3.9' --with 'nbformat>=5.10' --with 'nbclient>=0.10' \
  python run_notebook.py
```

`run_notebook.py` regenerates the `.ipynb` from the percent-cell source,
executes it, and fails on an assertion or notebook exception.  It does not
modify Prime-RL training code or register an environment.

From the Prime-RL repository root, the code checks are:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -q examples/opd_theory_toys
.venv/bin/ruff check examples/opd_theory_toys
```

The executed notebook is designed to finish in a few minutes on CPU; these
exact and low-dimensional experiments do not justify occupying a GPU node.
End-to-end LLM runs are explicitly left for a later phase; when that phase
starts, reuse `configs/debug/training_modes/{sft,opd,rl}.toml` and the existing
`reverse-text` environment rather than creating a new one.

## Interpretation guardrails

The benchmark `p_star` is a stipulated desired deployment law, not a claim
that every RL task has a unique, primitive true distribution.  The study keeps
`p_star`, the learned `p_theta`, `p_demo`, `q_teacher`, trusted utility, and
learned/noisy reward separate.  In a general KL-regularized RL problem, the
distributional target exists only after specifying a reference policy and KL
temperature:

`p_star(trajectory) proportional to p0(trajectory) exp(U_star / beta)`.

Changing `p0` or `beta` changes `p_star`.  Conversely, in the explicit-truth
mixture benchmark, entropy-calibrated RL receives the privileged reward
`log p_star`; teacher-only methods do not.  That information asymmetry is the
point of the teacher-mismatch test and must not be interpreted as an intrinsic
algorithmic advantage.

One-step toys are not used to make claims about prefix occupancy.  A sequential
toy is added only after the one-step controls pass and only for effects that a
one-step distribution cannot express.  See `DESIGN.md` for the gating rules.
