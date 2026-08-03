"""Detached-rollout OPD versus the full trajectory reverse-KL gradient.

Preregistered falsifier F4 in PREREGISTRATION.md.  Every practical OPD
implementation freezes the rollout distribution inside one update, which
drops the occupancy term of the trajectory reverse-KL derivative:

``grad KL(P_p || P_q) = g_semi + E[ grad log p(a_1) * c_{a_1} ]``,

where ``c_a`` is the downstream conditional reverse KL after first action
``a``.  In the smallest sequential model -- Bernoulli first step
``p = sigmoid(theta)``, frozen downstream conditionals with costs
``c_1, c_0`` -- both pieces are closed form:

- full derivative: ``p (1-p) [logit(p) - logit(q) + (c_1 - c_0)]``
- semigradient:    ``p (1-p) [logit(p) - logit(q)]``
- fixed points:    ``p*_full = sigmoid(logit(q) - (c_1 - c_0))`` versus
  ``p*_semi = q``
- opposite-direction region: ``0 < logit(q) - logit(p) < c_1 - c_0``.

The sampled estimators mirror real implementations: the semigradient path is
`toy_methods.opd_rkl_loss` per visited state; the occupancy correction is a
REINFORCE term whose reward is minus the downstream cost-to-go.  A
realizable control makes the downstream conditionals trainable, in which
case both vector fields share the teacher endpoint and only the path
differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from examples.opd_theory_toys.toy_methods import (
    RKLBatch,
    opd_rkl_loss,
    sample_categorical,
)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class TwoStepProblem:
    """First-step Bernoulli choice between two frozen downstream branches."""

    teacher_first_prob: float = 0.8
    branch1_student: tuple[float, ...] = (0.8, 0.1, 0.1)
    branch1_teacher: tuple[float, ...] = (0.1, 0.1, 0.8)
    branch0_student: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    branch0_teacher: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)

    def downstream_cost(self, branch: int) -> float:
        student = torch.tensor(
            self.branch1_student if branch == 1 else self.branch0_student, dtype=torch.float64
        )
        teacher = torch.tensor(
            self.branch1_teacher if branch == 1 else self.branch0_teacher, dtype=torch.float64
        )
        return float((student * (student / teacher).log()).sum())

    @property
    def cost_gap(self) -> float:
        return self.downstream_cost(1) - self.downstream_cost(0)

    def full_fixed_point(self) -> float:
        return 1.0 / (1.0 + math.exp(-(_logit(self.teacher_first_prob) - self.cost_gap)))

    def sign_disagreement(self, first_prob: float) -> bool:
        semi = _logit(first_prob) - _logit(self.teacher_first_prob)
        full = semi + self.cost_gap
        return semi * full < 0 or (semi == 0.0) != (full == 0.0)


def exact_gradients(problem: TwoStepProblem, first_prob: float) -> dict[str, float]:
    """d(loss)/d(theta) for the full trajectory RKL and its semigradient."""

    scale = first_prob * (1.0 - first_prob)
    semi = scale * (_logit(first_prob) - _logit(problem.teacher_first_prob))
    full = semi + scale * problem.cost_gap
    return {"semigradient": semi, "full": full}


def sampled_gradients(
    problem: TwoStepProblem,
    first_prob: float,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """One-batch estimates of both gradients, matching real estimators.

    The first-step action is sampled fresh (behavior = current policy).  The
    semigradient term is the Prime-like detached-advantage surrogate at the
    root state; the occupancy term multiplies the score by the detached
    downstream cost-to-go of the branch actually taken.
    """

    theta = torch.nn.Parameter(torch.tensor(_logit(first_prob), dtype=torch.float64))
    logits = torch.stack([torch.zeros((), dtype=torch.float64), theta])
    generator = torch.Generator().manual_seed(seed)
    rollout = sample_categorical(logits.detach(), batch_size, generator=generator)
    log_probs = logits.log_softmax(dim=-1)
    teacher_logp = torch.tensor(
        [1.0 - problem.teacher_first_prob, problem.teacher_first_prob], dtype=torch.float64
    ).log()
    student_logp = log_probs[rollout.action]
    semi_report = opd_rkl_loss(
        RKLBatch(
            student_logp=student_logp,
            behavior_logp=rollout.behavior_logp,
            teacher_logp=teacher_logp[rollout.action],
        )
    )
    semi_grad = torch.autograd.grad(semi_report.loss, theta, retain_graph=True)[0]
    costs = torch.tensor(
        [problem.downstream_cost(0), problem.downstream_cost(1)], dtype=torch.float64
    )
    importance_ratio = (student_logp - rollout.behavior_logp).exp()
    occupancy_loss = (costs[rollout.action].detach() * importance_ratio).mean()
    occupancy_grad = torch.autograd.grad(occupancy_loss, theta)[0]
    return {
        "semigradient": float(semi_grad),
        "full": float(semi_grad + occupancy_grad),
    }


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    first_prob: float


@dataclass(frozen=True)
class SemigradientTrajectory:
    update_rule: str
    initial_first_prob: float
    points: tuple[TrajectoryPoint, ...]

    @property
    def final_first_prob(self) -> float:
        return self.points[-1].first_prob


def run_trajectory(
    problem: TwoStepProblem,
    update_rule: str,
    *,
    initial_first_prob: float,
    steps: int = 4000,
    learning_rate: float = 0.05,
    record_every: int = 20,
) -> SemigradientTrajectory:
    theta = _logit(initial_first_prob)
    points = [TrajectoryPoint(step=0, first_prob=initial_first_prob)]
    for step in range(1, steps + 1):
        first_prob = 1.0 / (1.0 + math.exp(-theta))
        gradient = exact_gradients(problem, first_prob)[update_rule]
        theta -= learning_rate * gradient
        if step % record_every == 0 or step == steps:
            points.append(TrajectoryPoint(step=step, first_prob=1.0 / (1.0 + math.exp(-theta))))
    return SemigradientTrajectory(
        update_rule=update_rule,
        initial_first_prob=initial_first_prob,
        points=tuple(points),
    )


def sign_map_rows(
    problem: TwoStepProblem,
    *,
    first_probs: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20)),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for first_prob in first_probs:
        gradients = exact_gradients(problem, first_prob)
        rows.append(
            {
                "first_prob": first_prob,
                "semigradient": gradients["semigradient"],
                "full": gradients["full"],
                "opposite_directions": problem.sign_disagreement(first_prob),
                "predicted_boundary_low": problem.full_fixed_point(),
                "predicted_boundary_high": problem.teacher_first_prob,
            }
        )
    return rows


@dataclass(frozen=True)
class RealizableOutcome:
    update_rule: str
    final_first_prob: float
    final_branch1_kl: float
    path_area_versus_semi: float


def run_realizable_control(
    problem: TwoStepProblem,
    *,
    initial_first_prob: float = 0.6,
    steps: int = 20_000,
    learning_rate: float = 0.05,
) -> list[RealizableOutcome]:
    """Trainable downstream conditionals: endpoints must coincide.

    Both rules are run with exact per-state gradients.  The downstream
    branch-1 conditional trains toward the teacher under a forward-KL row
    update weighted by that state's occupancy (its visitation probability),
    which is how a frozen-rollout implementation would supervise it; the
    occupancy question only concerns the first-step parameter.
    """

    outcomes: list[RealizableOutcome] = []
    reference: list[float] | None = None
    for update_rule in ("semigradient", "full"):
        theta = _logit(initial_first_prob)
        branch1 = torch.tensor(problem.branch1_student, dtype=torch.float64).log().clone()
        branch1.requires_grad_(True)
        teacher1 = torch.tensor(problem.branch1_teacher, dtype=torch.float64)
        trajectory: list[float] = []
        for _ in range(steps):
            first_prob = 1.0 / (1.0 + math.exp(-theta))
            probs1 = branch1.softmax(dim=-1)
            with torch.no_grad():
                cost1 = float((probs1 * (probs1.log() - teacher1.log())).sum())
            scale = first_prob * (1.0 - first_prob)
            gradient = scale * (_logit(first_prob) - _logit(problem.teacher_first_prob))
            if update_rule == "full":
                gradient += scale * (cost1 - problem.downstream_cost(0))
            theta -= learning_rate * gradient
            row_loss = first_prob * (probs1 * (probs1.log() - teacher1.log())).sum()
            row_grad = torch.autograd.grad(row_loss, branch1)[0]
            with torch.no_grad():
                branch1 -= learning_rate * row_grad
            trajectory.append(first_prob)
        final_probs1 = branch1.detach().softmax(dim=-1)
        final_kl = float((final_probs1 * (final_probs1.log() - teacher1.log())).sum())
        if reference is None:
            reference = trajectory
            area = 0.0
        else:
            area = float(
                sum(abs(a - b) for a, b in zip(trajectory, reference)) / len(reference)
            )
        outcomes.append(
            RealizableOutcome(
                update_rule=update_rule,
                final_first_prob=1.0 / (1.0 + math.exp(-theta)),
                final_branch1_kl=final_kl,
                path_area_versus_semi=area,
            )
        )
    return outcomes
