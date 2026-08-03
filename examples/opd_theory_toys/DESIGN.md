# First-principles design: SFT, OPD, and RL

This document is the experiment contract.  The supplied `opd_theory.pdf`
contributed taxonomy and useful operator identities--support, occupancy, KL
direction, and teacher/reward mismatch--but it did not supply the experiment
template.  The benchmark families, controls, and branch rules below are
derived from the update operators and first principles.

## 1. Make `p_star` explicit and keep the information sources separate

For an input or prefix `s` and output/action `a`, distinguish:

1. `p_star(a|s)`: the stipulated desired/evaluation law;
2. `p_theta(a|s)`: the learned student law;
3. `d_demo(s), p_demo(a|s)`: demonstration states and actions;
4. `q_teacher(a|s)`: the teacher queried for probabilities;
5. `U_star(s,a)`: trusted deployment utility;
6. `R_hat(s,a)`: the possibly noisy or biased reward actually optimized.

The core comparison sets `p_demo = q_teacher`: demonstrations are unfiltered
teacher samples.  This makes SFT and teacher distillation share a target while
leaving `p_star` free to expose teacher omission or distortion.  Experiments
that relax this equality must say so explicitly.

`p_star` is a controlled benchmark object, not a claim that RL always has a
primitive true distribution.  In a general KL-regularized problem, a reference
policy `p0` and coefficient `beta` turn a utility into a distribution

`p_star(a|s) proportional to p0(a|s) exp(U_star(s,a) / beta)`.

Changing `p0` or `beta` changes that target.  We therefore report both
fidelity to the explicitly declared `p_star` and expected trusted utility.

## 2. Operators under test

At one state, with student `p_theta`, teacher `q_teacher`, and fresh on-policy
samples:

- SFT minimizes `E_{a~p_demo}[-log p_theta(a)]`.  Under the core equality
  `p_demo = q_teacher`, its population loss is
  `KL(q_teacher || p_theta)` plus a constant.
- OPD-FKL is the control `KL(q_teacher || p_theta)` evaluated on
  student-collected states.
  It is identical to soft offline KD in a one-state problem; only a sequential
  problem can make their state occupancies differ.
- Prime-like OPD-RKL samples `a~p_theta` and uses the detached advantage
  `log q_teacher(a)-log p_theta(a)`.  Its fresh-policy expected gradient is the
  gradient of `KL(p_theta || q_teacher)`.
- RL samples `a~p_theta` and follows a reward advantage.  With no fixed reference
  penalty it maximizes expected reward and generally does *not* reproduce a
  reward-shaped probability distribution.  A local trust region to the
  rollout policy is an optimizer constraint, not a persistent base-model
  regularizer.

The comparison records four independent axes: state occupancy, information
source, divergence/estimator, and cost.  A result is not called an "OPD
effect" if occupancy and KL direction changed simultaneously without the
factorial controls.

## 3. What the first toy must prove before becoming more complicated

### Stage 0: implementation and null controls

Use fully expressive categorical policies and exact expectations.

- When `p_demo = q_teacher = p_star`, SFT and both full-population
  distillation objectives must recover `p_star`; entropy/KL-calibrated RL must
  recover it when its target is explicitly calibrated to `p_star`.
- Pure unregularized RL is a negative control: with reward `log p_star`, it
  generally seeks a density peak rather than recovering the distribution.
- SFT/OPD-FKL must be identical in a one-state problem.
- Enumerated sampled OPD-RKL and RL gradients must match their analytic
  population gradients and the real Prime-RL losses.

Failure means an implementation or comparison bug; stop before interpreting
training curves.

### Stage 1: one-step distribution geometry

Use an ordered categorical grid.  Gaussian, shifted Gaussian, and two/three
mode mixtures are just normalized masses on that grid.  Cross:

- realizable free logits vs a capacity-limited one-bump student;
- mode separation and unequal mode weights;
- teacher/utility alignment;
- exact gradients vs finite sampled signals;
- support floor and initialization.

Predictions to falsify:

1. A shifted, realizable single Gaussian is a null control: both KL directions
   share the same optimum.
2. With a one-bump student and a separated mixture, forward KL compromises or
   covers, while reverse KL chooses a mode.  With enough student modes this
   asymptotic difference disappears.
3. RL selects according to utility, not teacher mass.  If all valid modes have
   the same scalar reward, RL cannot identify the teacher's relative weights.
4. An important outcome with current probability `p_i` supplies only about
   `N p_i` on-policy observations.  SFT sees it at rate `N q_teacher_i`; exact
   teacher logits are a different information budget again.

#### Primary benchmark: decaying truth, biased teacher, and student

`truth_mixture_experiments.py` makes the three distributions directly
inspectable.  The truth is a categorical law on `n` named modes with
geometrically decaying weights `w_star_i`.  `q_teacher` can sharpen, flatten,
floor, or exactly zero those modal probabilities; `p_theta` has one trainable
logit per structurally reachable mode and configurable epsilon initialization.
In the core runs, `p_demo = q_teacher`.

Equal-width, separated Gaussian kernels render the named categories for plots;
they are not the optimized density.  An optional labeled joint model
`p(mode, coordinate)` adds one shared student translation and exact
equal-variance conditional-Gaussian location terms.  Teacher mass weights the
SFT/FKL location term, whereas current student mass weights RKL/RL.  These
simple formulas rely on observed mode identity and are not claimed for a
generic unlabeled overlapping mixture.

Exact categorical population objectives test endpoint predictions before
adding estimator noise:

1. In a realizable family, population SFT, OPD-FKL, and OPD-RKL target
   `q_teacher`, so they inherit its distortion or omission.
2. Pure RL with trusted reward `log p_star` maximizes expected truth log
   density and is mode-seeking; it does not recover `p_star`.
3. Entropy-calibrated RL with objective
   `E_{p_theta}[log p_star] + H(p_theta)` minimizes
   `KL(p_theta || p_star)` and targets `p_star` when it is realizable.
4. The RL methods receive privileged truth-derived information unavailable to
   teacher-only methods.  Their truth recovery under a biased teacher tests
   information source and reachability, not an unconditional superiority of
   RL.

The main notebook uses geometric decay and varies mode count, teacher
epsilon-omission or weight distortion, and initial student coverage.  Exact
categorical zeros are a
separate limit: if a reachable student mode has `q_teacher_i = 0`, reverse KL
is infinite and the code reports it unavailable rather than smoothing it; a
hard student capacity mask is structural and cannot be reopened by any
method.  The labeled translation control separately varies location
distortion under a one-translation bottleneck.  Report both KL directions to
`p_star` and `q_teacher`, named-mode masses, effective support, and expected
truth log weight.  This is a one-state population benchmark: it cannot
establish a prefix-occupancy or finite-sample discovery claim by itself.

`truth_sampling_experiments.py` supplies the separate finite-action-sampling
control on the same named modes.  Hard SFT draws from `q_teacher`; reverse-PD
and RL draw from `p_theta`; full-logit forward-PD is deterministic conditional
on the fixed input.  Repeated initial-gradient estimates are checked against
the shared population operators, and observed mode-hit rates are checked
against `1 - (1 - probability_i)^N`.  This diagnoses action exposure only;
prefix occupancy is still reserved for the sequential experiment.

`truth_reward_experiments.py` then holds the policy and truth fixed while
changing only the scalar reward semantics.  Besides the existing
`log p_star` utility, it draws one reference mode from `p_star` per group and
scores student actions by exact binary match or normalized squared coordinate
distance.  The expected utility vector and therefore the pure-RL endpoint are
derived before sampling.  Prime-style group-relative advantages use the
unnormalized within-group mean baseline: their expected fresh-policy direction
is `(G - 1) / G` times ordinary policy gradient, while a homogeneous-reward
group has exactly zero update.  Predicted reward-contrast probabilities and
sampled gradient statistics are compared at a fixed total completion budget.
This is the score-function core of a GRPO-like update, not a claim to reproduce
the full LLM trainer (clipping, reference KL, and sequence credit are absent).

A secondary capacity-limited stress test chooses a one-Gaussian student so
that the three operators have different predicted destinations, not just
differently colored loss curves.  Here `p_star = q_teacher = p_demo`:

`p_star = 0.8 Normal(-4, 1) + 0.2 Normal(+4, 0.1)` with a one-Gaussian
student.

- FKL moment-matches: `mean=-2.4`, `variance=11.042`.
- RKL selects the broad left component because it has greater *probability
  mass* (`-log 0.8 < -log 0.2`).
- pure RL with utility `log p_star` collapses toward the narrow right component
  because it has the higher *density peak*.
- for `E_{p_theta} log p_star + alpha H(p_theta)`, separated-component
  asymptotics predict a right-to-left switch at
  `alpha = 1 - log(0.2/0.8) / log(0.1/1) = 0.397940...`.

This yields a reusable distinction: forward KL tracks coverage/moments,
reverse KL tracks representable probability mass, and unregularized
log-density RL tracks peak utility.  These statements are conditional on the
one-mode bottleneck and are checked again with a realizable family.

Branch rule: if exact objectives agree but sampled runs differ, the next sweep
targets estimator variance/support.  If exact optima differ, the next sweep
targets capacity and utility metrics.  Hard zeros are replaced by an epsilon
floor and epsilon is swept.

### Stage 1b: graded shift versus overlap-only support

For this aligned information-shape ablation, set
`p_demo = q_teacher = p_star` and use the same fixed-width Gaussian student for
two equally remote targets:

- a shifted Gaussian teacher, whose log-density tail supplies an exact
  location signal `displacement / teacher_variance` at every student sample;
- a target interval plus a flat epsilon background, whose OPD signal is
  `log(q_inside/q_outside) * d chi/d mean` and whose binary-RL signal is only
  `d chi/d mean`, where `chi` is current target-interval mass.

The finite-sample discovery prediction is
`P(hit) = 1 - (1 - chi)^N`, with the transition at `N chi ~= 1`.  Report raw
signal before KL matching: scaling each nonzero direction to the same KL
deliberately hides the probability and cost of obtaining that direction.  A
bridge is admitted only as an idealized reachability calculation until an
actual curriculum has a separate trusted final objective.

### Stage 2: smallest sequential occupancy test

Only after Stage 1 passes, use a two-step fork:

- teacher demonstrations occupy one second-step branch;
- the student also visits an error/recovery branch;
- teacher reliability on that branch is independently varied;
- reward may agree or disagree with the teacher.

Declare `p_star(a|s)` on both branches before running the comparison; teacher
reliability means agreement of `q_teacher` with that declared target, not merely
self-consistency.

Cross teacher-state/student-state collection with FKL/RKL.  This is the first
experiment allowed to make claims about *on-policy state occupancy*.
Increase horizon only if the two-step model cannot separate reachability from
recovery.

### Stage 3: curriculum and forgetting, conditional on an observed mechanism

A curriculum is tested only after a failure can be predicted from measured
support or reward contrast.  The fully expressive target-mismatch control uses
`p_demo = q_teacher` and is

`p_init=(1-2 epsilon, epsilon, epsilon)`,
`q_teacher=(.60,.35,.05)`, `U_star=(0,+1,-1)`

for neutral, desirable, and harmful actions.  Distillation has the stronger
support-acquisition signal but copies teacher error; RL has the desired
selector endpoint but requires reward contrast.  For distinct reward values,
group size `G`, and `B` groups,

`P(informative batch) = 1 - (sum_a p(a)^G)^B`.

Pure RL's distributional endpoint here is the point mass on the desirable
action.  If a non-degenerate `p_star` is required, declare an epsilon-smoothed
target or derive it from `p0`, `U_star`, and `beta` before evaluating KL.  Do
not infer relative desired frequencies from correctness rankings alone.

Switch from support acquisition to RL when the current-policy analytic rate
crosses a preregistered threshold, not at an arbitrary training step.  Keep
two comparisons separate:

- a common fixed step scale preserves raw signal magnitude and measures
  progress under a fixed call budget;
- KL-matching measures conditional direction geometry and must report the
  amplification scale, no-ops, and unused movement budget.

Teacher labels, scalar teacher log probabilities, full teacher vectors,
student rollouts, and reward calls remain separate cost coordinates.

For forgetting, use two contexts with controllable feature overlap.  Zero
overlap is the no-forgetting control.  Compare old-task retention at matched
new-task gain and cumulative behavioral KL, and give every method the same
replay/reference options.  A smaller raw update is not evidence that RL
intrinsically remembers better.  In the rank-one logistic null control,
`delta old logit = feature_overlap * delta new logit`; methods therefore have
identical forgetting at matched new progress even when their raw steps differ.

## 4. Comparison protocol

For every configuration:

1. derive or numerically enumerate the exact objective, optimum, and gradient;
2. make a prediction before sampling;
3. run finite-sample replications and show uncertainty, not one seed;
4. compare both equal calls and equal returned information where possible;
5. repeat with behavioral-KL-matched steps;
6. evaluate trusted utility, both KL directions to `p_star` and
   `q_teacher`, entropy, support/mode mass, and estimator signal-to-noise;
7. use the discrepancy between prediction and observation to choose the next
   experiment.

Costs are reported separately: teacher trajectories, teacher log-probability
values, student rollouts, reward calls, and optimizer updates.  There is no
single honest scalar conversion without an application-specific cost model.

## 5. Current implementation-level observation

The real Prime-RL loss checks pass for SFT, full-vocabulary OPD-RKL, sampled
OPD-RKL in expectation, and on-policy RL.  However, the current
teacher-top-k-plus-sampled OPD branch is not an unbiased sparse estimator of
full reverse KL.

Let `K` be teacher top-k, `b` the fixed rollout distribution, and
`h_i(theta)=p_theta(i) log[p_theta(i)/q_teacher(i)]`.  The expected sparse
scalar is

`sum_{i in K} h_i(theta) + sum_{i not in K} b_i h_i(theta)`.

At a fresh policy `b=p_theta`, omitted terms therefore receive another factor of
`p_i`; their value/gradient contribution is effectively quadratic in rare
student mass.  Exact enumeration against Prime-RL establishes two sharper
facts:

1. for omitted student mass `M` split across `m` symmetric outcomes, the
   aggregate tail direction changes sign at
   `(M/m)(1+r_tail) = 1+r_head`, while full RKL changes sign at
   `r_tail=r_head`; fragmentation therefore makes reversal more likely;
2. even `p_theta=q_teacher` is generally not stationary for
   `k < vocabulary_size`, because the per-token KL-gradient cancellation is
   broken by the unequal coefficients.

Importance weighting the sampled tail by `1/b_i` restores the exact mean but
can have extreme variance.  A deterministic residual bucket
`P_tail log(P_tail/Q_tail)` is a proper coarse KL and restores the teacher
fixed point, but deliberately loses within-tail allocation information.  This
is an exact statement about the current sparse branch; its practical effect
still has to be measured at realistic vocabularies and optimizer scales.
