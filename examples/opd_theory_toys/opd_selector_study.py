# %% [markdown]
# # Phase 2 — from mechanism atlas to a validated selector
#
# Read `PREREGISTRATION.md` first if you read nothing else; every claim below
# was written down there before any run.
#
# State of the roadmap (this notebook is built top-to-bottom as results land):
#
# 1. **Done** — preregistration of the three-headed selector + falsifiers F1-F6.
# 2. **Done** — E1a/E1a′: unlocking-rate falsifiers F1-F3 (F1, F2 confirmed;
#    F3 refuted, with the mechanism identified and a corrected claim).
# 3. **Done** — E1b: semigradient sign-flip falsifier F4 (confirmed exactly).
# 4. **Next** — depth-H two-track harness (F5), then the selector benchmark (F6).
#
# The frozen `opd_toy_study.ipynb` is the mechanism atlas; nothing here
# re-derives it.  This notebook exists to kill or confirm preregistered
# predictions and to end each section with a decision.

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cycler import cycler
from IPython.display import display

HERE = Path.cwd().resolve()
if not (HERE / "toy_methods.py").exists():
    matches = list(HERE.rglob("examples/opd_theory_toys/toy_methods.py"))
    if len(matches) != 1:
        raise RuntimeError("Run from the notebook directory or the Prime-RL checkout.")
    HERE = matches[0].parent
REPO = HERE.parents[1]
for path in (REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

torch.set_default_dtype(torch.float64)
torch.set_num_threads(min(4, os.cpu_count() or 1))
pd.set_option("display.precision", 4)
pd.set_option("display.width", 200)

PASTEL = {
    "ink": "#34495E",
    "muted": "#98A1B2",
    "blue": "#79A9DC",
    "mint": "#79C7A3",
    "sage": "#A8C982",
    "apricot": "#F1B56C",
    "coral": "#E8898F",
    "lavender": "#B59AD8",
    "teal": "#69B8B2",
    "rose": "#D97887",
    "paper": "#F7F8FC",
    "panel": "#FFFFFF",
    "grid": "#E5E8F0",
}
plt.rcParams.update(
    {
        "figure.figsize": (10.5, 5.6),
        "figure.dpi": 120,
        "figure.facecolor": PASTEL["paper"],
        "axes.facecolor": PASTEL["panel"],
        "axes.edgecolor": "#D6DAE5",
        "axes.labelcolor": PASTEL["ink"],
        "axes.titlecolor": PASTEL["ink"],
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(
            color=(
                PASTEL["blue"],
                PASTEL["apricot"],
                PASTEL["mint"],
                PASTEL["coral"],
                PASTEL["lavender"],
                PASTEL["teal"],
            )
        ),
        "grid.color": PASTEL["grid"],
        "grid.alpha": 0.8,
        "font.size": 12,
        "text.color": PASTEL["ink"],
        "lines.linewidth": 2.4,
        "legend.facecolor": PASTEL["panel"],
        "legend.edgecolor": "#DDE1EA",
        "legend.fontsize": 10,
    }
)

from examples.opd_theory_toys import semigradient_experiments as semi
from examples.opd_theory_toys import unlocking_experiments as unlock

print(f"repo {REPO}")

# %% [markdown]
# ## 0. What the selector must be, fixed in advance
#
# Three heads, because two provably cannot cover the phenomena the atlas
# already demonstrated:
#
# 1. **Whole-budget head**: $\mathrm{Gap}_m(C) = B_m + c_m V_m / 2C +
#    \mathrm{Opt}_m(C)$, with the $\mathrm{Opt}_m$ fitting protocol frozen in
#    `PREREGISTRATION.md` section 1 (anti-unfalsifiability rule).
# 2. **Next-block head**: local response $\eta a_m - \eta^2 b_m / 2$ per cost.
# 3. **Reachability head**: informative-update probability
#    $\pi_{batch}$ and discovery probability now and under each bridge —
#    because heads 1-2 are local and blind to support unlocking.
#
# Falsifiers F1-F6 and the selection-regret benchmark protocol (holdout by
# defect *family*, equal-cost-pilot baseline as the bar to beat) are in
# `PREREGISTRATION.md` sections 2-3.  Tonight's sections each open with the
# preregistered expectation and close with Saw / Verdict / Next.

# %% [markdown]
# ## 1. E1a — unlocking a rare correct token (falsifiers F1-F3)
#
# **Expected (preregistered).**
#
# 1. F1: with exact signals under SGD, steps-to-unlock scale like
#    $\log(1/p_0)$ for forward KL (slope $\approx 0$ on log-log) and
#    superlinearly for reverse KL (slope $\ge 0.7$).
# 2. F2: under Adam the separation collapses below $4\times$ at
#    $p_0 = 10^{-4}$; if it survives, the KL-direction story is
#    information-theoretic rather than an optimizer artifact.
# 3. F3: sampled reverse-KL cannot beat the discovery floor
#    $\sim 1/(N p_0)$ under either optimizer.
#
# Setup: one 16-token state, teacher puts 0.6 on the target token, student
# starts at $p_0 \in \{10^{-1},\dots,10^{-4}\}$, batch 64, thresholds and
# learning rates preregistered in `unlocking_experiments.py`.

# %%
bulk_study = unlock.run_unlock_study()
bulk_summary = pd.DataFrame(bulk_study.summary_rows())
bulk_scaling = pd.DataFrame(bulk_study.scaling_rows())
display(bulk_summary[bulk_summary.p0.isin((1e-1, 1e-4))])
display(bulk_scaling)

# %% [markdown]
# ### E1a′ — cold-target control, added after staring at E1a
#
# E1a's first-hit times beat the $1/(Np_0)$ floor by $5\times$: the student's
# bulk surplus mass *must* flow into the only under-weighted token, so
# softmax renormalization finds the target without ever sampling it.  That is
# a real mechanism, not a bug — but it means E1a cannot test F3.  This
# control removes it: the student equals the teacher away from a small-mass
# target ($q_t = 0.05$), so only a thin uniform surplus remains, and $p_0$
# extends to $10^{-6}$ where the naive floor predicts $\sim 15{,}600$
# batches.

# %%
cold_study = unlock.run_cold_unlock_study()
cold_summary = pd.DataFrame(cold_study.summary_rows())
cold_scaling = pd.DataFrame(cold_study.scaling_rows())
display(cold_summary[cold_summary.p0.isin((1e-4, 1e-6))])
display(cold_scaling)

# %%
estimator_colors = {
    "fkl_exact": PASTEL["mint"],
    "sft_sampled": PASTEL["sage"],
    "rkl_exact": PASTEL["apricot"],
    "rkl_sampled": PASTEL["coral"],
}
optimizer_styles = {"sgd": "-", "adam": "--"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for axis, (title, summary, scaling) in zip(
    axes,
    (
        ("Bulk-surplus regime (E1a)", bulk_summary, bulk_scaling),
        ("Cold-target regime (E1a′)", cold_summary, cold_scaling),
    ),
    strict=True,
):
    for estimator, color in estimator_colors.items():
        for optimizer_name, line_style in optimizer_styles.items():
            group = summary[
                (summary.estimator == estimator)
                & (summary.optimizer == optimizer_name)
                & (summary.censored == 0)
            ].sort_values("p0")
            if group.empty:
                continue
            slope_rows = scaling[
                (scaling.estimator == estimator) & (scaling.optimizer == optimizer_name)
            ]
            slope = float(slope_rows.scaling_exponent.iloc[0])
            axis.plot(
                1.0 / group.p0,
                group.median_steps,
                line_style,
                marker="o",
                ms=5,
                color=color,
                label=f"{estimator} / {optimizer_name} (slope {slope:.2f})",
            )
    axis.set(xscale="log", yscale="log", xlabel="$1/p_0$", title=title)
    axis.legend(fontsize=8.5)
axes[0].set_ylabel("median steps to threshold")
fig.suptitle("Reverse-KL unlocking cost is an SGD phenomenon; Adam flattens every curve", y=1.03)
fig.tight_layout(pad=1.5)
plt.show()

# %% [markdown]
# ### E1a + E1a′ — Saw / Verdict / Next
#
# **Saw.**
#
# 1. F1 confirmed: SGD-exact slopes 0.70 and 0.73 (reverse KL, bulk and cold)
#    versus 0.13-0.15 (forward KL).  At $p_0=10^{-4}$ the bulk-regime gap is
#    2,237 vs 37 steps — a $60\times$ separation.
# 2. F2 confirmed beyond the preregistered bar: under Adam the separation
#    *inverts* (exact RKL 68 steps vs exact FKL 101 at $p_0=10^{-4}$);
#    cold-regime slopes 0.17 vs 0.21.  The unlocking penalty is an optimizer
#    artifact, not information geometry.
# 3. F3 **refuted twice**.  Bulk regime: first hits at ~34 batches versus the
#    ~156 floor (renormalization mass flow).  Cold regime: median unlock
#    stays ~165 steps from $p_0=10^{-4}$ to $10^{-6}$ under Adam (floor says
#    15,600) — softmax score functions leak gradient into never-sampled
#    logits, and Adam amplifies the thin consistent surplus to lr-sized
#    steps.  Under SGD the barrier is real but is the vector field, not
#    sampling: sampled ≈ exact (26,966 vs 25,514 steps at $10^{-5}$).
#
# **Verdict.**  Token-level support barriers at visited states do not
# survive adaptive optimizers.  The honest support story is at the *state*
# level (unvisited prefixes have exactly zero gradient and are
# unidentifiable — the atlas two-step result stands).  Scope note: LLMs
# share parameters across states, so the per-logit leak channel is
# entangled there; this kills the clean theory claim, not necessarily every
# practical instance of it.
#
# **Next.**  Reachability head of the selector must track state visitation
# and reward contrast, not token discovery.  Depth-H harness (F5) becomes
# the load-bearing support experiment.

# %% [markdown]
# ## 2. E1b — detached-rollout OPD vs the full trajectory gradient (F4)
#
# **Expected (preregistered).**  With first-step probability $p$, teacher
# $q=0.8$, and downstream reverse-KL costs $c_1 - c_0 = 1.456$ frozen:
# semigradient fixed point at $p^*=q$, full-gradient fixed point at
# $\sigma(\mathrm{logit}\,q - (c_1-c_0)) = 0.483$, opposite update
# directions exactly on $p \in (0.483, 0.8)$, unbiased sampled estimators,
# and no endpoint gap once the downstream conditionals are trainable
# (realizable control).

# %%
problem = semi.TwoStepProblem()
sign_rows = pd.DataFrame(semi.sign_map_rows(problem))
display(
    pd.DataFrame(
        [
            {
                "cost gap c1-c0": problem.cost_gap,
                "semigradient fixed point": problem.teacher_first_prob,
                "full-gradient fixed point": problem.full_fixed_point(),
                "measured flip region": (
                    float(sign_rows[sign_rows.opposite_directions].first_prob.min()),
                    float(sign_rows[sign_rows.opposite_directions].first_prob.max()),
                ),
            }
        ]
    )
)

estimator_rows = []
for first_prob in (0.3, 0.6):
    exact = semi.exact_gradients(problem, first_prob)
    estimates = {"semigradient": [], "full": []}
    for seed in range(400):
        sampled = semi.sampled_gradients(problem, first_prob, batch_size=64, seed=seed)
        for key in estimates:
            estimates[key].append(sampled[key])
    for key, values in estimates.items():
        mean = float(np.mean(values))
        se = float(np.std(values) / np.sqrt(len(values)))
        estimator_rows.append(
            {
                "first_prob": first_prob,
                "estimator": key,
                "exact": exact[key],
                "sampled_mean": mean,
                "z": (mean - exact[key]) / se,
            }
        )
display(pd.DataFrame(estimator_rows))

trajectories = {
    rule: semi.run_trajectory(problem, rule, initial_first_prob=0.6)
    for rule in ("semigradient", "full")
}
realizable = semi.run_realizable_control(problem)
display(pd.DataFrame([outcome.__dict__ for outcome in realizable]))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
rule_colors = {"semigradient": PASTEL["apricot"], "full": PASTEL["blue"]}
for rule, trajectory in trajectories.items():
    axes[0].plot(
        [point.step for point in trajectory.points],
        [point.first_prob for point in trajectory.points],
        color=rule_colors[rule],
        label=f"{rule} (ends {trajectory.final_first_prob:.3f})",
    )
axes[0].axhline(problem.teacher_first_prob, color=PASTEL["apricot"], ls=":", lw=1.4)
axes[0].axhline(problem.full_fixed_point(), color=PASTEL["blue"], ls=":", lw=1.4)
axes[0].set(
    xlabel="step",
    ylabel="first-step probability",
    title="Same checkpoint, opposite directions (init 0.6)",
)
axes[0].legend()

probs_grid = sign_rows.first_prob
axes[1].axhline(0.0, color=PASTEL["ink"], lw=1.2)
axes[1].plot(probs_grid, sign_rows.semigradient, color=PASTEL["apricot"], label="semigradient")
axes[1].plot(probs_grid, sign_rows.full, color=PASTEL["blue"], label="full gradient")
axes[1].axvspan(
    problem.full_fixed_point(),
    problem.teacher_first_prob,
    color=PASTEL["rose"],
    alpha=0.12,
    label="predicted flip region",
)
axes[1].set(
    xlabel="first-step probability $p$",
    ylabel="$dL/d\\theta$",
    title="Sign disagreement exactly where predicted",
)
axes[1].legend()
fig.suptitle("F4: frozen-rollout OPD and trajectory reverse KL optimize different vector fields", y=1.03)
fig.tight_layout(pad=1.5)
plt.show()

# %% [markdown]
# ### E1b — Saw / Verdict / Next
#
# **Saw.**
#
# 1. Sign flip confirmed: from $p=0.6$ the semigradient converges to 0.8000,
#    the full gradient to 0.4827; both match closed forms to $10^{-4}$.
# 2. Measured flip region $[0.5, 0.8]$ on a 0.05 grid = predicted
#    $(0.483, 0.8)$ at grid resolution.
# 3. Sampled estimators unbiased over 400 batches ($|z| < 1.7$).
# 4. Realizable control: endpoints coincide (both 0.8000, downstream KL
#    $\sim 10^{-16}$), mean path gap 0.0008 — the divergence is a
#    capacity-limited phenomenon only.
#
# **Verdict.**  F4 confirmed exactly.  Detached-rollout OPD over-commits the
# student to branches it cannot imitate downstream; full trajectory reverse
# KL trades marginal fidelity for downstream imitability.  Which is "right"
# depends on the evaluation: trajectory-level quality favors the full
# gradient, per-state imitation favors the semigradient.  Under capacity
# limits — the LLM regime — they are different methods and should be named
# separately.
#
# **Next.**  The depth-H harness should log both vector fields' fixed
# points; the selector's whole-budget head must use the *semigradient*
# endpoint for OPD bias $B_m$, because that is what implementations run.

# %% [markdown]
# ## 3. E2 — depth-H two-track process: compounding, recovery cost, budgets (F5)
#
# **Expected (preregistered).**
#
# 1. Cumulative metric: SFT regret follows the ceiling law
#    $\varepsilon H \min(H, u)$ with untrained excursion
#    $u = 1/(\pi_0 \rho_{rec})$; on-policy teacher-labeled training stays
#    $\sim \varepsilon H \cdot u_{opd}$ with small $u_{opd}$.
# 2. The OPD advantage must vanish when $u = \Theta(H)$
#    ($\rho_{rec} \sim 1/H$) and when recovery is impossible, and must not
#    appear on the bounded terminal metric.
# 3. Sparse terminal-reward RL hits a discovery wall when
#    $p_{succ} = (1-\varepsilon_0)^H$ makes informative groups vanish;
#    dense process RL does not.
# 4. Realized on-policy disagreement is logged so a failed separation is
#    attributable (per the rebuttal's requirement).

# %%
from examples.opd_theory_toys import depth_h_experiments as depth

analytic_df = pd.DataFrame(depth.analytic_gap_rows())
ceiling_view = analytic_df[(analytic_df.recover_spec == "1.0") & (analytic_df.pi0.isin((0.5, 0.01)))]
display(
    ceiling_view.pivot_table(
        index="H", columns="pi0", values=["sft_J_cum", "opd_J_cum"]
    ).round(3)
)

fig, axis = plt.subplots(figsize=(8.5, 5.4))
for pi0, color in ((0.5, PASTEL["mint"]), (0.1, PASTEL["apricot"]), (0.01, PASTEL["coral"])):
    group = analytic_df[
        (analytic_df.recover_spec == "1.0") & (analytic_df.pi0 == pi0)
    ].sort_values("H")
    axis.plot(
        group.H,
        group.sft_J_cum,
        "o-",
        color=color,
        label=f"SFT-like, $\\pi_0$={pi0} (u={1/pi0:.0f})",
    )
opd_group = analytic_df[
    (analytic_df.recover_spec == "1.0") & (analytic_df.pi0 == 0.5)
].sort_values("H")
axis.plot(opd_group.H, opd_group.opd_J_cum, "s--", color=PASTEL["blue"], label="OPD-like")
axis.set(
    xscale="log",
    yscale="log",
    xlabel="horizon $H$",
    ylabel="exact cumulative off-track cost",
    title="Not bare $H^2$: a ceiling law $\\varepsilon H \\min(H, u)$",
)
axis.legend()
fig.tight_layout(pad=1.5)
plt.show()

# %% [markdown]
# The analytic part already sharpens the folklore: SFT's quadratic regime is
# a *transient* that ends at $H \approx u$; what separates the methods
# asymptotically is the ratio of off-track excursion lengths, not the
# exponent.  Both DP values match the absorbing closed form
# $\sum_t 1-(1-\varepsilon)^{t-1}$ to $10^{-12}$.

# %%
trained = depth.run_trained_study()
trained_df = pd.DataFrame([outcome.__dict__ for outcome in trained])
trained_agg = trained_df.groupby(["horizon", "recover_prob", "method"], as_index=False).agg(
    J_cum=("J_cum", "median"),
    J_term=("J_term", "median"),
    on_err=("on_track_error", "median"),
    off_err=("off_track_error", "median"),
    info_rate=("informative_batch_rate", "median"),
)
display(trained_agg[trained_agg.horizon.isin((8, 32))].round(4))

# %%
perfect_rows = []
for horizon in (4, 8, 16, 32):
    for spec in (1.0, "1/H", 0.0):
        recover_prob = 1.0 / horizon if spec == "1/H" else float(spec)
        problem = depth.TwoTrackProblem(
            horizon=horizon, recover_prob=recover_prob, teacher_correct_prob=1.0
        )
        for method in ("sft", "opd_fkl"):
            for seed in range(4):
                outcome = depth.train_method(method, problem, seed=seed)
                perfect_rows.append(
                    {
                        "H": horizon,
                        "spec": str(spec),
                        "method": method,
                        "J_cum": outcome.J_cum,
                        "on_err": outcome.on_track_error,
                        "off_err": outcome.off_track_error,
                    }
                )
perfect_df = (
    pd.DataFrame(perfect_rows)
    .groupby(["H", "spec", "method"], as_index=False)
    .median()
)
ratio_df = perfect_df.pivot_table(index=["H", "spec"], columns="method", values="J_cum")
ratio_df["sft_over_opd"] = ratio_df["sft"] / ratio_df["opd_fkl"]
display(ratio_df.round(3))

# Epsilon-controlled DP replay: equalize on-track error across methods and
# re-evaluate, so the ratio isolates the recovery channel.
controlled_rows = []
for horizon in (4, 8, 16, 32):
    for spec in (1.0, "1/H", 0.0):
        recover_prob = 1.0 / horizon if spec == "1/H" else float(spec)
        problem = depth.TwoTrackProblem(horizon=horizon, recover_prob=recover_prob)
        shared_on = torch.full((horizon,), 1.0 - 0.015, dtype=torch.float64)
        sft_eval = depth.evaluate_policy(
            problem, shared_on, torch.full((horizon,), 0.1, dtype=torch.float64)
        )
        opd_eval = depth.evaluate_policy(
            problem, shared_on, torch.full((horizon,), 0.62, dtype=torch.float64)
        )
        controlled_rows.append(
            {
                "H": horizon,
                "spec": str(spec),
                "controlled_ratio": sft_eval["J_cum"] / max(opd_eval["J_cum"], 1e-12),
            }
        )
controlled_df = pd.DataFrame(controlled_rows).pivot_table(
    index="H", columns="spec", values="controlled_ratio"
)
display(controlled_df.round(3))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
spec_colors = {"1.0": PASTEL["mint"], "1/H": PASTEL["apricot"], "0.0": PASTEL["coral"]}
for spec, color in spec_colors.items():
    measured = ratio_df.reset_index()
    group = measured[measured.spec == spec].sort_values("H")
    axes[0].plot(group.H, group.sft_over_opd, "o-", color=color, label=f"measured, recover={spec}")
    control = controlled_df[spec]
    axes[0].plot(control.index, control.values, ":", color=color, lw=1.8)
axes[0].axhline(1.0, color=PASTEL["ink"], lw=1.2)
axes[0].set(
    xscale="log",
    xlabel="horizon $H$",
    ylabel="J_cum ratio SFT / OPD",
    title="Recovery advantage grows with H, collapses when recovery is hard\n(dotted: $\\varepsilon$-controlled DP replay)",
)
axes[0].legend(fontsize=9)

wall = trained_agg[(trained_agg.method == "rl_sparse") & (trained_agg.recover_prob == 0.0)]
axes[1].plot(wall.horizon, wall.info_rate, "o-", color=PASTEL["coral"], label="informative-batch rate")
axes[1].plot(wall.horizon, wall.J_term, "s--", color=PASTEL["lavender"], label="terminal success")
axes[1].set(
    xscale="log",
    xlabel="horizon $H$",
    ylabel="rate",
    title="Sparse-RL discovery wall (irreversible track)",
    ylim=(-0.04, 1.04),
)
axes[1].legend()
fig.tight_layout(pad=1.5)
plt.show()

# %% [markdown]
# ### E2 — Saw / Verdict / Next
#
# **Saw.**
#
# 1. Ceiling law confirmed exactly in the DP and in trained policies: SFT's
#    "quadratic compounding" ends at $H \approx u$; the asymptotic
#    separation is the excursion-length ratio.  With a perfect teacher the
#    trained ratio grows 1.88 → 8.70 over $H = 4 \to 32$ at easy recovery.
# 2. F5 collapse confirmed after $\varepsilon$-control: at
#    $\rho_{rec} = 1/H$ and $0$, the $\varepsilon$-controlled replay ratios
#    sit near 1 while raw ratios (1.3-2.8) are explained by an on-track
#    error mismatch between the methods (0.029 vs 0.009-0.014, cause not yet
#    identified — logged as an open micro-question, it does not affect the
#    recovery conclusion because the dotted controls isolate it).
# 3. Sparse-RL discovery wall: informative-batch rate 0.95 → 0.70 → 0.00 →
#    0.00 for $H = 4,8,16,32$ at irreversible recovery; at $H \ge 16$ the
#    policy receives literally zero learning signal for the entire budget.
# 4. Two unplanned findings.  (a) Metric-objective alignment: dense
#    process-RL wins $J_{cum}$ but collapses $J_{term}$ (it rationally
#    tolerates late falls); sparse terminal-RL does the reverse.  (b) A
#    realistic teacher (2% error) leaks recovery supervision into demos
#    ($\sim 38H$ off-track demo steps per budget here), compressing the
#    SFT/OPD gap: **a slightly clumsy teacher inoculates SFT against
#    compounding**.  The DAgger advantage is proportional to how error-free
#    the demonstrations are.
#
# **Verdict.**  F5 confirmed in its honest form: the on-policy advantage is
# a recovery-channel effect with a ceiling, not a universal $H^2$ vs $H$
# law; it disappears exactly where predicted and never appears on the
# terminal metric.
#
# **Next.**  The selector benchmark (F6): the reachability head gets the
# informative-batch predictor validated here; $B_m$ for OPD uses the
# semigradient endpoint (E1b); "which metric" is now an explicit input, not
# an afterthought.

# %% [markdown]
# ## 4. E3 — crossover maps and held-out selection regret (F6)
#
# **Expected (preregistered).**
#
# 1. The three-headed selector beats always-SFT/OPD/RL, random, and the
#    equal-cost pilot on mean selection regret over held-out defect
#    *families* (leave-one-family-out).  Losing to any always-one-method
#    baseline = rejection.
# 2. Winner structure follows the phase geometry: small teacher-truth gap →
#    distillation; large gap + adequate reachability → RL/handoff; RL's
#    share grows with budget.
# 3. Ablation rule: if removing the fitted $\kappa C^{-\rho}$ term changes
#    held-out regret by more than 20% relative, head 1 is declared
#    unvalidated.
#
# Tonight's selector uses oracle features (exact endpoints, exact
# reachability); the deployable-probe version is phase 3.  Grid: 4 defect
# families × 3 sizes × 3 init supports × 4 rewards × 2 seeds = 288 configs,
# 5 methods each, trusted loss $L^* = TV(p_\theta, p^*)$ recorded at budgets
# $\{3, 6, 12, 25, 50\}$.

# %%
from examples.opd_theory_toys import selector_benchmark as bench

grid = bench.build_grid()
benchmark_rows = bench.run_benchmark(grid)
bench_df = pd.DataFrame(benchmark_rows)
method_df = bench_df[bench_df.method.isin(bench.METHODS)]
winners = method_df.loc[
    method_df.groupby(
        ["family", "size_index", "epsilon", "reward_kind", "seed", "budget"]
    ).trusted_loss.idxmin()
]
display(winners.groupby(["budget", "method"]).size().unstack(fill_value=0))
final_winners = winners[winners.budget == 50].copy()
display(final_winners.groupby(["family", "method"]).size().unstack(fill_value=0))
final_winners["gap_bin"] = pd.qcut(
    final_winners.teacher_gap, 3, labels=["small gap", "mid gap", "large gap"]
)
display(
    final_winners.groupby(["gap_bin", "method"], observed=True).size().unstack(fill_value=0)
)

# %%
regret_df = pd.DataFrame(bench.evaluate_selectors(benchmark_rows))
regret_summary = (
    regret_df.groupby("strategy")
    .agg(mean_regret=("regret", "mean"), p90=("regret", lambda s: s.quantile(0.9)))
    .sort_values("mean_regret")
)
display(regret_summary.round(4))
regret_by_budget = regret_df.pivot_table(
    index="strategy", columns="budget", values="regret", aggfunc="mean"
)
display(regret_by_budget.round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
strategy_styles = {
    "three_headed": (PASTEL["coral"], "-", 3.0),
    "ablate_fitted_term": (PASTEL["rose"], ":", 1.8),
    "ablate_reachability": (PASTEL["lavender"], ":", 1.8),
    "always_rl": (PASTEL["blue"], "--", 2.0),
    "always_opd_rkl": (PASTEL["apricot"], "--", 2.0),
    "always_sft": (PASTEL["mint"], "--", 2.0),
    "pilot_baseline": (PASTEL["ink"], "-.", 2.0),
    "random": (PASTEL["muted"], "-", 1.5),
}
for strategy, (color, line_style, width) in strategy_styles.items():
    if strategy not in regret_by_budget.index:
        continue
    series = regret_by_budget.loc[strategy]
    axes[0].plot(
        series.index, series.values, line_style, color=color, lw=width, label=strategy
    )
axes[0].set(
    xscale="log",
    xlabel="budget (updates)",
    ylabel="mean held-out selection regret",
    title="Selector beats every baseline at every budget",
)
axes[0].legend(fontsize=8.5)

share = (
    winners.groupby(["budget", "method"]).size().unstack(fill_value=0)
)
share = share.div(share.sum(axis=1), axis=0)
bottom = np.zeros(len(share))
method_colors = {
    "sft": PASTEL["mint"],
    "opd_fkl": PASTEL["sage"],
    "opd_rkl": PASTEL["apricot"],
    "rl": PASTEL["coral"],
    "handoff": PASTEL["lavender"],
}
for method in bench.METHODS:
    axes[1].bar(
        range(len(share)),
        share[method].values,
        bottom=bottom,
        color=method_colors[method],
        label=method,
        width=0.72,
    )
    bottom += share[method].values
axes[1].set_xticks(range(len(share)), share.index)
axes[1].set(
    xlabel="budget (updates)",
    ylabel="share of configs won",
    title="RL's share grows with budget; distillation owns the short game",
)
axes[1].legend(fontsize=9)
fig.tight_layout(pad=1.5)
plt.show()

# %% [markdown]
# ### E3 — Saw / Verdict / Next
#
# **Saw.**
#
# 1. F6 passed: three-headed selector mean held-out regret 0.026 vs 0.035
#    for the best baseline (always-RL), and it is the column minimum at all
#    five budgets.  Exact-pick rate 31%.
# 2. Ablations pass the preregistered rule: dropping the fitted term moves
#    regret ~12% relative (< 20%), so bias + reachability carry the
#    structure; the fitted term doubles the exact-pick rate (15% → 31%).
#    Dropping the reachability filter costs ~5%.
# 3. The equal-cost pilot — flagged in advance as "the real bar" — is the
#    *worst* strategy at budget 50 (regret 0.158, below random 0.142):
#    two-update probes anti-predict the long-run winner because fast
#    distillation starts mask slow RL finishes.  Theory-based selection is
#    not merely competitive with brute-force probing; probing is actively
#    misleading at realistic probe costs.
# 4. Phase geometry as predicted: at budget 50, small teacher-truth gap →
#    OPD-RKL (59/96 wins), large gap → RL + handoff (67/96).  Family
#    fingerprints: `sharpen` → OPD-RKL sweeps (70/72); `headmiss` → OPD-RKL
#    wins **zero** (mode-seeking cannot restore a missing dominant mode) and
#    RL/handoff/FKL split it.
#
# **Verdict.**  The when-to-use-what question has a validated answer on
# this grid: endpoint bias + reachability + a budget-dependent estimator
# term, fit on other defect families, out-predicts every baseline —
# including brute-force pilots.  Caveats: oracle features, one-state
# problems, TV as the declared trusted loss, uniform price vector.
#
# **Next (phase 3, not tonight).**  Deployable-probe features (estimate
# $B_m$ and reachability from finite queries, charge them); sequential
# configs from the depth-H harness in the grid; the three preregistered
# price vectors; head 2 as tie-break among distillation variants.

# %% [markdown]
# ## Running log — end of night
#
# 1. F1 ✓ (SGD reverse-KL unlocking slope 0.70/0.73 vs forward 0.13).
# 2. F2 ✓ and stronger than preregistered: Adam *inverts* the gap; the
#    KL-direction unlocking penalty is an optimizer artifact.
# 3. F3 ✗ refuted — softmax renormalization + Adam defeat token-level
#    sample absence; support is a *state-level* phenomenon.  Corrected
#    claim recorded in section 1.
# 4. F4 ✓ exact: frozen-rollout OPD and trajectory reverse KL are different
#    methods under capacity limits (opposite directions in the predicted
#    region; identical endpoints when realizable).
# 5. F5 ✓ as a ceiling law $\varepsilon H \min(H, u)$, not bare $H^2$ vs
#    $H$; collapse at $u = \Theta(H)$ confirmed after $\varepsilon$-control;
#    sparse-RL discovery wall at $H \ge 16$; two unplanned findings
#    (metric-objective alignment; teacher error inoculates SFT).
# 6. F6 ✓ first-ever run of the selector benchmark: preregistered selector
#    beats all baselines on held-out defect families at every budget, and
#    the equal-cost pilot is anti-predictive at long budgets.
