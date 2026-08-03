"""Selector benchmark: crossover maps and held-out selection regret (F6).

Protocol preregistered in PREREGISTRATION.md section 3.  One-state
categorical mode problems: decaying truth ``p*``, defective teacher ``q_T``,
support-limited student init.  Every method trains once per config with the
trusted loss ``L* = TV(p_theta, p*)`` recorded at all budget checkpoints, so
selection strategies are evaluated offline against the same result table.

Defect families (the holdout unit):
- ``omission``: teacher keeps only the top-j truth modes.
- ``headmiss``: teacher omits the largest mode(s).
- ``sharpen``: teacher weights proportional to ``p*^power``.
- ``shift``: teacher mass partially cyclic-shifted one mode over.

Methods: ``sft`` (teacher samples), ``opd_fkl`` (full teacher row),
``opd_rkl`` (Prime-like sampled reverse KL), ``rl`` (batch-centered policy
gradient on the config's reward), and ``handoff`` (SFT until the analytic
informative-batch probability crosses 0.9, then RL).

The preregistered selector is oracle-featured tonight (exact endpoints and
reachability from the config): head 1 predicts ``Gap_m(C) = B_m +
kappa_m C^-rho_m`` with ``kappa, rho`` fit on training families only; head 3
filters methods whose informative-batch probability at init is below 0.5
with no bridge.  Baselines: always-<method>, random, and the equal-cost
pilot (first ~20% of updates split round-robin, remainder committed to the
best pilot improvement; charged honestly by giving it fewer remaining
updates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from examples.opd_theory_toys.toy_methods import (
    RKLBatch,
    RLBatch,
    SFTBatch,
    opd_rkl_loss,
    rl_loss,
    sample_categorical,
    sft_loss,
)

MODES = 5
DECAY = 0.55
BATCH = 32
CHECKPOINTS = (3, 6, 12, 25, 50)
METHODS = ("sft", "opd_fkl", "opd_rkl", "rl", "handoff")
FAMILIES = ("omission", "headmiss", "sharpen", "shift")
VALID_THRESHOLD = 0.05


@dataclass(frozen=True)
class ModeConfig:
    family: str
    size_index: int
    epsilon: float
    reward_kind: str
    seed: int

    @property
    def key(self) -> tuple[object, ...]:
        return (self.family, self.size_index, self.epsilon, self.reward_kind, self.seed)


def truth_probs() -> Tensor:
    weights = torch.tensor([DECAY**i for i in range(MODES)], dtype=torch.float64)
    return weights / weights.sum()


def teacher_probs(config: ModeConfig) -> Tensor:
    truth = truth_probs()
    floor = 1e-5
    if config.family == "omission":
        keep = (3, 2, 1)[config.size_index]
        probs = truth.clone()
        probs[keep:] = floor
    elif config.family == "headmiss":
        drop = (1, 2, 3)[config.size_index]
        probs = truth.clone()
        probs[:drop] = floor
    elif config.family == "sharpen":
        power = (1.5, 2.0, 3.0)[config.size_index]
        probs = truth**power
    elif config.family == "shift":
        share = (0.25, 0.5, 1.0)[config.size_index]
        probs = (1.0 - share) * truth + share * torch.roll(truth, 1)
    else:
        raise ValueError(f"unknown family {config.family!r}")
    return probs / probs.sum()


def initial_logits(config: ModeConfig) -> Tensor:
    probs = torch.full((MODES,), config.epsilon, dtype=torch.float64)
    probs[MODES - 1] = 1.0 - (MODES - 1) * config.epsilon
    return probs.log()


def reward_values(config: ModeConfig) -> Tensor:
    truth = truth_probs()
    if config.reward_kind.startswith("binary"):
        return (truth >= VALID_THRESHOLD).to(torch.float64)
    if config.reward_kind == "distance":
        best = int(truth.argmax())
        indices = torch.arange(MODES, dtype=torch.float64)
        return -((indices - best) / 1.0) ** 2 / 4.0
    if config.reward_kind == "logp":
        return truth.log()
    raise ValueError(f"unknown reward {config.reward_kind!r}")


def reward_noise_std(config: ModeConfig) -> float:
    return 0.5 if config.reward_kind.endswith("noisy") else 0.0


def trusted_loss(student_probs: Tensor) -> float:
    return float(0.5 * (student_probs - truth_probs()).abs().sum())


def informative_batch_probability(student_probs: Tensor, config: ModeConfig) -> float:
    """P(a size-BATCH group spans >= 2 reward classes) for the config reward."""

    values = reward_values(config)
    if reward_noise_std(config) > 0:
        return 1.0  # continuous noise always produces contrast
    classes: dict[float, float] = {}
    for mode in range(MODES):
        classes.setdefault(round(float(values[mode]), 12), 0.0)
        classes[round(float(values[mode]), 12)] += float(student_probs[mode])
    return 1.0 - sum(mass**BATCH for mass in classes.values())


def method_endpoint_bias(method: str, config: ModeConfig) -> float:
    """Head 1's ``B_m``: trusted loss of the method's population endpoint."""

    if method in ("sft", "opd_fkl", "opd_rkl"):
        return trusted_loss(teacher_probs(config))
    values = reward_values(config)
    best = values == values.max()
    endpoint = best.to(torch.float64) / best.sum()
    return trusted_loss(endpoint)


def run_config_method(
    config: ModeConfig,
    method: str,
    *,
    updates: int = max(CHECKPOINTS),
    learning_rate: float = 0.05,
    pilot_plan: tuple[int, str] | None = None,
) -> dict[int, float]:
    """Train once, return {checkpoint: trusted loss}.

    ``pilot_plan`` (used by the pilot baseline) runs ``pilot_plan[0]`` update
    rounds cycling through all base methods before committing to
    ``pilot_plan[1]`` for the remainder.
    """

    method_salt = sum(ord(ch) for ch in method)
    generator = torch.Generator().manual_seed(config.seed * 9973 + method_salt)
    logits = torch.nn.Parameter(initial_logits(config))
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    teacher = teacher_probs(config)
    values = reward_values(config)
    noise_std = reward_noise_std(config)
    losses: dict[int, float] = {}
    switched = method != "handoff"
    base_cycle = ("sft", "opd_fkl", "opd_rkl", "rl")

    for step in range(updates):
        if pilot_plan is not None:
            active = (
                base_cycle[step % len(base_cycle)] if step < pilot_plan[0] else pilot_plan[1]
            )
        elif method == "handoff":
            if not switched:
                probs = logits.detach().softmax(dim=-1)
                if informative_batch_probability(probs, config) >= 0.9:
                    switched = True
            active = "rl" if switched else "sft"
        else:
            active = method
        optimizer.zero_grad(set_to_none=True)
        log_probs = logits.log_softmax(dim=-1)
        if active == "sft":
            demo = sample_categorical(teacher.log(), BATCH, generator=generator)
            report = sft_loss(SFTBatch(student_logp_demo=log_probs[demo.action]))
        elif active == "opd_fkl":
            report_loss = -(teacher * log_probs).sum()
            report_loss.backward()
            optimizer.step()
            probs = logits.detach().softmax(dim=-1)
            if step + 1 in CHECKPOINTS:
                losses[step + 1] = trusted_loss(probs)
            continue
        elif active == "opd_rkl":
            rollout = sample_categorical(logits.detach(), BATCH, generator=generator)
            report = opd_rkl_loss(
                RKLBatch(
                    student_logp=log_probs[rollout.action],
                    behavior_logp=rollout.behavior_logp,
                    teacher_logp=teacher.log()[rollout.action],
                )
            )
        elif active == "rl":
            rollout = sample_categorical(logits.detach(), BATCH, generator=generator)
            reward = values[rollout.action]
            if noise_std > 0:
                reward = reward + noise_std * torch.randn(
                    reward.shape, generator=generator, dtype=torch.float64
                )
            advantage = reward - reward.mean()
            report = rl_loss(
                RLBatch(
                    student_logp=log_probs[rollout.action],
                    behavior_logp=rollout.behavior_logp,
                    advantage=advantage,
                )
            )
        else:
            raise ValueError(f"unknown method {active!r}")
        report.loss.backward()
        optimizer.step()
        if step + 1 in CHECKPOINTS:
            losses[step + 1] = trusted_loss(logits.detach().softmax(dim=-1))
    return losses


def build_grid(
    *,
    epsilons: tuple[float, ...] = (1e-1, 1e-2, 5e-4),
    reward_kinds: tuple[str, ...] = ("binary", "binary_noisy", "distance", "logp"),
    seeds: tuple[int, ...] = (0, 1),
) -> list[ModeConfig]:
    configs = []
    for family in FAMILIES:
        for size_index in range(3):
            for epsilon in epsilons:
                for reward_kind in reward_kinds:
                    for seed in seeds:
                        configs.append(
                            ModeConfig(family, size_index, epsilon, reward_kind, seed)
                        )
    return configs


def run_benchmark(configs: list[ModeConfig]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config in configs:
        init_probs = initial_logits(config).softmax(dim=-1)
        reachability = informative_batch_probability(init_probs, config)
        for method in METHODS:
            losses = run_config_method(config, method)
            for checkpoint, loss_value in losses.items():
                rows.append(
                    {
                        "family": config.family,
                        "size_index": config.size_index,
                        "epsilon": config.epsilon,
                        "reward_kind": config.reward_kind,
                        "seed": config.seed,
                        "method": method,
                        "budget": checkpoint,
                        "trusted_loss": loss_value,
                        "B_m": method_endpoint_bias(method, config),
                        "init_reachability": reachability,
                        "teacher_gap": trusted_loss(teacher_probs(config)),
                    }
                )
        pilot_updates = {3: 4, 6: 4, 12: 4, 25: 8, 50: 8}
        for budget in CHECKPOINTS:
            pilot_rounds = pilot_updates[budget]
            if pilot_rounds >= budget:
                choice = "sft"
            else:
                probe_losses = {}
                for method in ("sft", "opd_fkl", "opd_rkl", "rl"):
                    probe = run_config_method(
                        config, method, updates=max(1, pilot_rounds // 4)
                    )
                    probe_losses[method] = min(probe.values()) if probe else math.inf
                choice = min(probe_losses, key=probe_losses.get)
            pilot = run_config_method(
                config, "pilot", updates=budget, pilot_plan=(pilot_rounds, choice)
            )
            rows.append(
                {
                    "family": config.family,
                    "size_index": config.size_index,
                    "epsilon": config.epsilon,
                    "reward_kind": config.reward_kind,
                    "seed": config.seed,
                    "method": "pilot_baseline",
                    "budget": budget,
                    "trusted_loss": pilot.get(budget, math.nan),
                    "B_m": math.nan,
                    "init_reachability": reachability,
                    "teacher_gap": trusted_loss(teacher_probs(config)),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Offline selection strategies evaluated against the result table.


def fit_head1(train_rows: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    """Fit ``Gap_m(C) - B_m ~ kappa C^-rho`` per method on training rows."""

    fits: dict[str, tuple[float, float]] = {}
    for method in METHODS:
        points = [
            (float(row["budget"]), float(row["trusted_loss"]) - float(row["B_m"]))
            for row in train_rows
            if row["method"] == method and float(row["trusted_loss"]) > float(row["B_m"]) + 1e-4
        ]
        if len(points) < 8:
            fits[method] = (0.0, 1.0)
            continue
        xs = np.log([p[0] for p in points])
        ys = np.log([p[1] for p in points])
        slope, intercept = np.polyfit(xs, ys, 1)
        fits[method] = (float(np.exp(intercept)), float(-slope))
    return fits


def select_three_headed(
    config_rows: list[dict[str, object]],
    fits: dict[str, tuple[float, float]],
    budget: int,
    *,
    use_fitted_term: bool = True,
    use_reachability: bool = True,
) -> str:
    """Preregistered selector; the keyword flags are the mandated ablations."""

    reachability = float(config_rows[0]["init_reachability"])
    best_method, best_score = None, math.inf
    for method in METHODS:
        method_rows = [r for r in config_rows if r["method"] == method]
        if not method_rows:
            continue
        bias = float(method_rows[0]["B_m"])
        if use_reachability and method == "rl" and reachability < 0.5:
            score = math.inf  # head 3: no usable signal, no bridge
        elif use_fitted_term:
            kappa, rho = fits[method]
            score = bias + kappa * budget ** (-rho)
        else:
            score = bias
        if score < best_score:
            best_method, best_score = method, score
    return best_method or "sft"


def evaluate_selectors(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Leave-one-family-out regret for the selector and all baselines."""

    results: list[dict[str, object]] = []
    method_rows = [r for r in rows if r["method"] in METHODS]
    for held_out in FAMILIES:
        train = [r for r in method_rows if r["family"] != held_out]
        test_configs: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in method_rows:
            if row["family"] == held_out:
                key = (
                    row["family"],
                    row["size_index"],
                    row["epsilon"],
                    row["reward_kind"],
                    row["seed"],
                )
                test_configs.setdefault(key, []).append(row)
        fits = fit_head1(train)
        rng = np.random.default_rng(0)
        for key, config_rows in test_configs.items():
            for budget in CHECKPOINTS:
                budget_rows = {
                    r["method"]: float(r["trusted_loss"])
                    for r in config_rows
                    if r["budget"] == budget
                }
                if len(budget_rows) < len(METHODS):
                    continue
                oracle = min(budget_rows.values())
                pilot_loss = next(
                    (
                        float(r["trusted_loss"])
                        for r in rows
                        if r["method"] == "pilot_baseline"
                        and r["budget"] == budget
                        and (
                            r["family"],
                            r["size_index"],
                            r["epsilon"],
                            r["reward_kind"],
                            r["seed"],
                        )
                        == key
                    ),
                    math.nan,
                )
                strategies = {
                    "three_headed": budget_rows[
                        select_three_headed(config_rows, fits, budget)
                    ],
                    "ablate_fitted_term": budget_rows[
                        select_three_headed(
                            config_rows, fits, budget, use_fitted_term=False
                        )
                    ],
                    "ablate_reachability": budget_rows[
                        select_three_headed(
                            config_rows, fits, budget, use_reachability=False
                        )
                    ],
                    "random": budget_rows[rng.choice(METHODS)],
                    "pilot_baseline": pilot_loss,
                }
                for method in METHODS:
                    strategies[f"always_{method}"] = budget_rows[method]
                for strategy, loss_value in strategies.items():
                    results.append(
                        {
                            "held_out_family": held_out,
                            "budget": budget,
                            "strategy": strategy,
                            "regret": loss_value - oracle,
                        }
                    )
    return results
