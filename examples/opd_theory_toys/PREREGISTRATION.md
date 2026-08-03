# Phase-2 preregistration: selector definitions, falsifiers, evaluation protocol

Written 2026-08-03, before any phase-2 experiment was run. The frozen
`opd_toy_study.ipynb` is the mechanism atlas; this file fixes what phase 2
must predict, on what grid, against which baselines, so that later analysis
cannot quietly move the goalposts.

## 1. The selector has three heads, not one

For candidate method `m` at checkpoint `theta` with remaining budget `C`
(reported per resource, section 4):

1. **Whole-budget head** — predicted trusted gap after spending `C`:
   `Gap_m(C) = B_m + c_m * V_m / (2C) + Opt_m(C)`.
   `B_m` = endpoint bias of `m`'s population target measured by the declared
   trusted loss; `V_m` = sandwich variance of `m`'s estimator (for sparse
   grouped RL use the exact group-moment formulas, never a dense oracle);
   `c_m` = cost per effective observation.
2. **Next-block head** — local response
   `dJ_m(eta) = eta * a_m - eta^2 * b_m / 2`, `a_m = grad J . u_m`,
   `b_m = u_m^T C_J u_m + tr(C_J V_m)`, reported as a conservative
   lower-bound gain per declared cost.
3. **Reachability head** — informative-update probability now and its
   predicted trajectory under each candidate bridge:
   `pi_group(p) = 1 - sum_v c_v(p)^G`, `pi_batch = 1 - (1-pi_group)^M`,
   plus discovery probability `1-(1-chi)^N` for support targets.
   This head exists because heads 1-2 are local and provably blind to
   support unlocking; a selector without it must fail configs whose binding
   constraint is reachability.

**Opt_m(C) protocol (anti-unfalsifiability rule).** `Opt_m` is fit as a
two-parameter curve `Opt_m(C) = kappa_m * C^(-rho_m)` on TRAINING-family
pilot runs only, frozen before touching holdout families. If the fitted
`Opt_m` term is doing most of the predictive work (ablating it changes
held-out regret by more than 20% relative), the whole-budget head is
declared unvalidated regardless of headline regret.

## 2. Preregistered falsifiers (each can kill a claim tonight)

- **F1 (unlocking scaling, SGD + exact signals).** Steps to lift a rare
  correct token from `p0` to 0.5: FKL scales like `log(1/p0)`; exact RKL
  scales superlinearly in `1/p0` (theory: `1/(p0 log(1/p0))`). Measured
  log-log slope for RKL >= 0.7 across `p0 in {1e-1..1e-4}`, FKL slope ~ 0.
- **F2 (optimizer dependence).** Under Adam with exact signals, the F1
  separation collapses to a small constant factor (< 4x at `p0 = 1e-4`,
  versus > 100x predicted under SGD). If the gap survives Adam, the
  KL-direction story is information-theoretic, not an optimizer artifact —
  a major revision to our own critique.
- **F3 (sample absence beats optimizer choice).** Sampled RKL (prime-like,
  batch N) cannot beat the discovery floor `~1/(N p0)` under either
  optimizer; sampled SFT stays `p0`-independent.
- **F4 (semigradient sign flip).** Detached-rollout OPD (the thing every
  real implementation runs) and full trajectory reverse-KL move the
  first-step probability in OPPOSITE directions whenever
  `c1 - c0 > logit(q) - logit(p)`, with fixed points `q` versus
  `sigma(logit q - (c1 - c0))`. Exact region boundary must match sampled
  runs; in the realizable control the discrepancy must vanish at the
  endpoint.
- **F5 (depth-H compounding, later tonight).** Cumulative recoverable-error
  metric: SFT regret grows ~ `eps * H^2`, on-policy teacher-labeled
  training ~ `eps * H`; the OPD advantage must DISAPPEAR when recovery
  cost-to-go is made Theta(H) (irreversible tracks) and on the single
  terminal-reward metric. Log on-policy disagreement eps_pi separately so a
  failed separation is attributable.
- **F6 (selector regret, the main event).** On held-out defect FAMILIES
  (not held-out grid cells), the three-headed selector must beat every
  baseline in section 3 on mean selection regret. Beating all except the
  equal-cost-pilot baseline = partial validation (theory describes but does
  not out-predict brute force). Losing to always-one-method = rejection.

## 3. Selector benchmark protocol (run only after F1-F5 are digested)

1. Config grid: 150-250 structural configs over teacher defect
   {omission, head-miss, sharpening, translation} x defect size x initial
   student support {epsilon in 1e-1..5e-4} x reward
   {oracle-log-density, binary, distance; exact and noisy} x budget
   C in {3, 6, 12, 25, 50} x 2 paired seeds. Capacity: full categorical
   (capacity-limited families deferred to a later phase).
2. Holdout: entire defect families rotate out (leave-one-family-out), plus
   an extrapolation split (largest defect sizes held out within families).
3. Methods selectable: SFT, OPD-FKL, OPD-RKL, RL(binary), RL(distance),
   and SFT->RL handoff at the reachability-head switch rule.
4. Metric: selection regret
   `r(x, C) = E[L*(chosen m, x, C)] - min_m E[L*(m, x, C)]`,
   reported as mean and 90th percentile over held-out configs, per budget.
5. Baselines: always-SFT, always-OPD, always-RL, random, cheapest-method,
   and equal-cost-pilot (spend 15% of C running all methods briefly, pick
   the best trusted gain per cost; the 15% is charged). No
   "pick-by-initial-loss" baseline: raw losses of different objectives are
   not comparable.

## 4. Cost accounting

No universal call exchange rate. Every result reports four meters
separately: teacher hard labels, teacher log-prob/full-vector queries,
student rollouts, reward calls. Any scalarization uses one of three
preregistered price vectors (teacher-expensive, reward-expensive, uniform)
and all three must agree on a claimed winner or the result is reported as
price-dependent.

## 5. Fairness rules inherited from the atlas

Fixed common scale and KL-matched comparisons reported separately; no-op
updates counted, never silently dropped; same optimizer family and seeds
across methods within a config; float64; population endpoints verified
against closed forms before finite-sample runs are trusted.
