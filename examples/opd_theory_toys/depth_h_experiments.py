"""Depth-H two-track process: compounding error, recovery cost, and budgets.

Preregistered falsifier F5 in PREREGISTRATION.md.  The environment is the
smallest process where state occupancy, recovery, and horizon interact:

- States ``(t, track)`` for steps ``t = 1..H``, track in {on, off}; start
  on-track.  Action 0 is correct/recover, action 1 leaves the track.
- On-track: action 0 stays on, action 1 falls off.  Off-track: action 0
  returns on-track with dynamics probability ``recover_prob`` (the recovery
  difficulty knob: 1 is easy, ``1/H`` makes the expected off-track excursion
  Theta(H), 0 is irreversible), action 1 stays off.
- Cumulative metric ``J_cum`` = expected number of off-track steps (expert
  value 0).  Terminal metric ``J_term`` = probability of ending on-track
  (expert value 1).

Part A (analytic): per-state-class policies are set directly and evaluated
by exact forward dynamic programming -- no sampling anywhere.  An SFT-like
policy has on-track error ``eps`` and an UNTRAINED off-track policy (its
demonstrations never visit off-track states); an OPD-like policy has error
``eps`` on both classes.  The predicted structure is a ceiling law, not a
bare H-squared: with untrained recovery probability ``pi0`` and dynamics
``recover_prob``, the expected off-track excursion is
``u = 1/(pi0 * recover_prob)`` and ``J_cum ~ eps * H * min(H, u)``.  The
on-policy advantage must vanish when ``u`` reaches Theta(H) for both
policies (recover_prob ~ 1/H) and on the bounded terminal metric.

Part B (trained): SFT, on-policy distillation (DAgger-style forward KL on
student-visited states), dense process-reward RL, and sparse terminal-reward
group-centered RL are actually trained from matched trajectory budgets, then
evaluated exactly by the same DP.  Realized per-class on-policy
disagreement is logged so a failed separation is attributable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from examples.opd_theory_toys.toy_methods import (
    FKLBatch,
    RLBatch,
    SFTBatch,
    opd_fkl_loss,
    rl_loss,
    sft_loss,
)


@dataclass(frozen=True)
class TwoTrackProblem:
    horizon: int = 8
    recover_prob: float = 1.0
    teacher_correct_prob: float = 0.98


def evaluate_policy(
    problem: TwoTrackProblem,
    on_correct: Tensor,
    off_recover: Tensor,
) -> dict[str, float]:
    """Exact DP evaluation of per-step policies.

    ``on_correct[t]`` is P(action 0 | on-track at step t); ``off_recover[t]``
    is P(action 0 | off-track at step t).  Returns exact cumulative off-track
    cost and terminal on-track probability.
    """

    horizon = problem.horizon
    if on_correct.shape != (horizon,) or off_recover.shape != (horizon,):
        raise ValueError("policy tensors must have shape (horizon,)")
    on_mass = 1.0
    off_mass = 0.0
    cumulative_off = 0.0
    for t in range(horizon):
        cumulative_off += off_mass
        stay_on = on_mass * float(on_correct[t])
        fall_off = on_mass * (1.0 - float(on_correct[t]))
        recovered = off_mass * float(off_recover[t]) * problem.recover_prob
        on_mass, off_mass = stay_on + recovered, fall_off + (off_mass - recovered)
    return {
        "J_cum": cumulative_off,
        "J_term": on_mass,
        "final_off_mass": off_mass,
    }


def analytic_gap_rows(
    *,
    horizons: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
    eps: float = 0.02,
    untrained_recover_probs: tuple[float, ...] = (0.5, 0.1, 0.01),
    recover_probs: tuple[float | str, ...] = (1.0, 0.25, "1/H", 0.0),
) -> list[dict[str, object]]:
    """Part A: exact ``J_cum``/``J_term`` for SFT-like vs OPD-like policies."""

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        for recover_spec in recover_probs:
            recover_prob = 1.0 / horizon if recover_spec == "1/H" else float(recover_spec)
            problem = TwoTrackProblem(horizon=horizon, recover_prob=recover_prob)
            on_policy_correct = torch.full((horizon,), 1.0 - eps, dtype=torch.float64)
            opd_like = evaluate_policy(
                problem, on_policy_correct, torch.full((horizon,), 1.0 - eps, dtype=torch.float64)
            )
            for pi0 in untrained_recover_probs:
                sft_like = evaluate_policy(
                    problem, on_policy_correct, torch.full((horizon,), pi0, dtype=torch.float64)
                )
                excursion = (
                    float("inf")
                    if pi0 * recover_prob == 0.0
                    else 1.0 / (pi0 * recover_prob)
                )
                rows.append(
                    {
                        "H": horizon,
                        "eps": eps,
                        "recover_prob": recover_prob,
                        "recover_spec": str(recover_spec),
                        "pi0": pi0,
                        "predicted_excursion_u": excursion,
                        "sft_J_cum": sft_like["J_cum"],
                        "opd_J_cum": opd_like["J_cum"],
                        "cum_advantage": sft_like["J_cum"] - opd_like["J_cum"],
                        "sft_J_term": sft_like["J_term"],
                        "opd_J_term": opd_like["J_term"],
                        "term_advantage": opd_like["J_term"] - sft_like["J_term"],
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Part B: trained methods.


@dataclass(frozen=True)
class TrainedOutcome:
    method: str
    horizon: int
    recover_prob: float
    seed: int
    budget_trajectories: int
    J_cum: float
    J_term: float
    on_track_error: float
    off_track_error: float
    off_state_visits: float
    informative_batch_rate: float | None
    cost_demo_trajectories: int
    cost_student_rollouts: int
    cost_teacher_rows: int
    cost_reward_calls: int


class TabularPolicy(torch.nn.Module):
    def __init__(self, horizon: int, *, initial_off_recover: float = 0.5) -> None:
        super().__init__()
        logits = torch.zeros((horizon, 2, 2), dtype=torch.float64)
        logits[:, 1, 0] = torch.log(torch.tensor(initial_off_recover, dtype=torch.float64))
        logits[:, 1, 1] = torch.log(torch.tensor(1.0 - initial_off_recover, dtype=torch.float64))
        self.logits = torch.nn.Parameter(logits)

    def action_probs(self) -> Tensor:
        return self.logits.softmax(dim=-1)

    def correct_probs(self) -> tuple[Tensor, Tensor]:
        probs = self.action_probs()
        return probs[:, 0, 0].detach(), probs[:, 1, 0].detach()


def _rollout(
    policy_probs: Tensor,
    problem: TwoTrackProblem,
    batch: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Vectorized student rollouts.  track: 0 = on, 1 = off."""

    horizon = problem.horizon
    tracks = torch.zeros((batch, horizon), dtype=torch.long)
    actions = torch.zeros((batch, horizon), dtype=torch.long)
    current = torch.zeros((batch,), dtype=torch.long)
    for t in range(horizon):
        tracks[:, t] = current
        p_correct = policy_probs[t, current, 0]
        take_correct = torch.rand((batch,), generator=generator, dtype=torch.float64) < p_correct
        actions[:, t] = torch.where(take_correct, 0, 1)
        on_now = current == 0
        fall = on_now & ~take_correct
        recover_roll = (
            torch.rand((batch,), generator=generator, dtype=torch.float64) < problem.recover_prob
        )
        recover = (~on_now) & take_correct & recover_roll
        current = torch.where(fall, 1, current)
        current = torch.where(recover, 0, current)
    off_counts = (tracks == 1).sum(dim=1)
    terminal_on = current == 0
    return {
        "tracks": tracks,
        "actions": actions,
        "off_counts": off_counts,
        "terminal_on": terminal_on,
    }


def train_method(
    method: str,
    problem: TwoTrackProblem,
    *,
    updates: int = 60,
    batch: int = 32,
    learning_rate: float = 0.05,
    seed: int = 0,
    group_size: int = 8,
    initial_off_recover: float = 0.1,
) -> TrainedOutcome:
    """Train one method at a matched trajectory budget and evaluate exactly.

    Budget: every method consumes ``updates * batch`` trajectories.  SFT
    consumes teacher demonstrations; the others consume student rollouts.
    Costs are metered separately and never merged.
    """

    generator = torch.Generator().manual_seed(seed)
    policy = TabularPolicy(problem.horizon, initial_off_recover=initial_off_recover)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    teacher_row = torch.tensor(
        [problem.teacher_correct_prob, 1.0 - problem.teacher_correct_prob], dtype=torch.float64
    )
    off_state_visits = 0
    informative_batches = 0
    cost_demo = cost_rollout = cost_rows = cost_reward = 0

    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        if method == "sft":
            demo_correct = (
                torch.rand((batch, problem.horizon), generator=generator, dtype=torch.float64)
                < problem.teacher_correct_prob
            )
            # Teacher rollouts: an error drops the teacher off-track; it
            # recovers with the same dynamics, so demos are overwhelmingly
            # on-track states with action 0.
            log_probs = policy.logits.log_softmax(dim=-1)
            current = torch.zeros((batch,), dtype=torch.long)
            demo_logps = []
            for t in range(problem.horizon):
                action = torch.where(demo_correct[:, t], 0, 1)
                demo_logps.append(log_probs[t, current, action])
                fall = (current == 0) & (action == 1)
                recover_roll = (
                    torch.rand((batch,), generator=generator, dtype=torch.float64)
                    < problem.recover_prob
                )
                recover = (current == 1) & (action == 0) & recover_roll
                current = torch.where(fall, 1, current)
                current = torch.where(recover, 0, current)
            report = sft_loss(SFTBatch(student_logp_demo=torch.stack(demo_logps, dim=1)))
            cost_demo += batch
        else:
            with torch.no_grad():
                rollout = _rollout(policy.action_probs(), problem, batch, generator)
            tracks, actions = rollout["tracks"], rollout["actions"]
            off_state_visits += int((tracks == 1).sum())
            cost_rollout += batch
            time_index = torch.arange(problem.horizon).expand(batch, -1)
            log_probs = policy.logits.log_softmax(dim=-1)
            if method == "opd_fkl":
                student_rows = log_probs[time_index, tracks, :]
                report = opd_fkl_loss(
                    FKLBatch(
                        student_log_probs=student_rows,
                        teacher_probs=teacher_row.expand_as(student_rows),
                    )
                )
                cost_rows += batch * problem.horizon
            elif method in ("rl_dense", "rl_sparse"):
                student_logp = log_probs[time_index, tracks, actions]
                behavior_logp = student_logp.detach()
                if method == "rl_dense":
                    off_indicator = (tracks == 1).to(torch.float64)
                    future_cost = off_indicator.flip(1).cumsum(1).flip(1)
                    advantage = -(future_cost - future_cost.mean(dim=0, keepdim=True))
                else:
                    reward = rollout["terminal_on"].to(torch.float64)
                    groups = reward.reshape(-1, group_size)
                    centered = groups - groups.mean(dim=1, keepdim=True)
                    if bool((centered.abs().sum(dim=1) > 0).any()):
                        informative_batches += 1
                    advantage = centered.reshape(-1, 1).expand(-1, problem.horizon)
                report = rl_loss(
                    RLBatch(
                        student_logp=student_logp,
                        behavior_logp=behavior_logp,
                        advantage=advantage,
                    )
                )
                cost_reward += batch
            else:
                raise ValueError(f"unknown method {method!r}")
        report.loss.backward()
        optimizer.step()

    on_correct, off_recover = policy.correct_probs()
    evaluation = evaluate_policy(problem, on_correct, off_recover)
    occupancy_on = _on_track_occupancy(problem, on_correct, off_recover)
    on_error = float(
        ((1.0 - on_correct) * occupancy_on).sum() / occupancy_on.sum().clamp_min(1e-12)
    )
    return TrainedOutcome(
        method=method,
        horizon=problem.horizon,
        recover_prob=problem.recover_prob,
        seed=seed,
        budget_trajectories=updates * batch,
        J_cum=evaluation["J_cum"],
        J_term=evaluation["J_term"],
        on_track_error=on_error,
        off_track_error=float(1.0 - off_recover.mean()),
        off_state_visits=float(off_state_visits),
        informative_batch_rate=(
            informative_batches / updates if method == "rl_sparse" else None
        ),
        cost_demo_trajectories=cost_demo,
        cost_student_rollouts=cost_rollout,
        cost_teacher_rows=cost_rows,
        cost_reward_calls=cost_reward,
    )


def _on_track_occupancy(
    problem: TwoTrackProblem, on_correct: Tensor, off_recover: Tensor
) -> Tensor:
    on_mass = torch.zeros((problem.horizon,), dtype=torch.float64)
    current_on, current_off = 1.0, 0.0
    for t in range(problem.horizon):
        on_mass[t] = current_on
        stay = current_on * float(on_correct[t])
        recovered = current_off * float(off_recover[t]) * problem.recover_prob
        fallen = current_on * (1.0 - float(on_correct[t]))
        current_on = stay + recovered
        current_off = fallen + (current_off - recovered)
    return on_mass


def run_trained_study(
    *,
    horizons: tuple[int, ...] = (4, 8, 16, 32),
    recover_probs: tuple[float | str, ...] = (1.0, "1/H", 0.0),
    methods: tuple[str, ...] = ("sft", "opd_fkl", "rl_dense", "rl_sparse"),
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    updates: int = 60,
    batch: int = 32,
) -> list[TrainedOutcome]:
    outcomes: list[TrainedOutcome] = []
    for horizon in horizons:
        for recover_spec in recover_probs:
            recover_prob = 1.0 / horizon if recover_spec == "1/H" else float(recover_spec)
            problem = TwoTrackProblem(horizon=horizon, recover_prob=recover_prob)
            for method in methods:
                for seed in seeds:
                    outcomes.append(
                        train_method(
                            method,
                            problem,
                            seed=seed,
                            updates=updates,
                            batch=batch,
                        )
                    )
    return outcomes
