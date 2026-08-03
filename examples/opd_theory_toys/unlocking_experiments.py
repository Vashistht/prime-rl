"""Rare-token unlocking rates: KL direction x optimizer x information source.

Preregistered falsifiers F1-F3 in PREREGISTRATION.md.  One categorical state;
the teacher puts order-one mass on a target token the student initially gives
probability ``p0``.  The question is how the number of optimizer steps needed
to lift the target token to a fixed threshold scales with ``1/p0``, and how
much of that scaling is (a) the KL direction, (b) the optimizer's coordinate
normalization, (c) sample absence.

Conditions
----------
- ``fkl_exact``: full-teacher-row cross entropy (population forward KL).
- ``rkl_exact``: analytic ``KL(p || q)`` (population reverse KL).
- ``rkl_sampled``: Prime-like detached-advantage estimator on ``N`` fresh
  student samples per step (`toy_methods.opd_rkl_loss`).
- ``sft_sampled``: hard cross entropy on ``N`` fresh teacher samples per step.

Each condition runs under plain SGD and under Adam with preregistered
learning rates.  Scaling exponents are slopes of ``log10(steps)`` against
``log10(1/p0)``; gradient-flow theory for the exact conditions predicts
slope ~0 for forward KL and slope ~1 (minus a log correction) for reverse KL
under SGD in logit coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from examples.opd_theory_toys.toy_methods import (
    FKLBatch,
    RKLBatch,
    SFTBatch,
    opd_fkl_loss,
    opd_rkl_loss,
    sample_categorical,
    sft_loss,
)

Estimator = str
OPTIMIZER_LEARNING_RATES: dict[str, float] = {"sgd": 0.5, "adam": 0.05}
EXACT_ESTIMATORS = ("fkl_exact", "rkl_exact")
SAMPLED_ESTIMATORS = ("sft_sampled", "rkl_sampled")


@dataclass(frozen=True)
class UnlockProblem:
    """One categorical state with a rare target token.

    ``student_matched_to_teacher = False`` is the bulk-surplus regime: the
    student is uniform away from the target, so every non-target token is
    heavily over-weighted and softmax renormalization alone must push large
    mass into the target.  ``True`` is the cold-target regime: the student
    equals the teacher conditioned on not-target (each non-target token is
    over-weighted only by the thin factor ``(1-p0)/(1-q_t)``), which removes
    the bulk mass-flow channel and exposes the sampling floor.
    """

    vocab_size: int = 16
    target_index: int = 0
    teacher_target_prob: float = 0.6
    initial_target_prob: float = 1e-2
    student_matched_to_teacher: bool = False

    def teacher_probs(self) -> Tensor:
        probs = torch.full(
            (self.vocab_size,),
            (1.0 - self.teacher_target_prob) / (self.vocab_size - 1),
            dtype=torch.float64,
        )
        probs[self.target_index] = self.teacher_target_prob
        return probs

    def initial_logits(self) -> Tensor:
        if self.student_matched_to_teacher:
            probs = self.teacher_probs() * (1.0 - self.initial_target_prob) / (
                1.0 - self.teacher_target_prob
            )
        else:
            probs = torch.full(
                (self.vocab_size,),
                (1.0 - self.initial_target_prob) / (self.vocab_size - 1),
                dtype=torch.float64,
            )
        probs[self.target_index] = self.initial_target_prob
        return probs.log()


@dataclass(frozen=True)
class UnlockRun:
    estimator: str
    optimizer: str
    initial_target_prob: float
    seed: int | None
    steps_to_threshold: int | None
    first_hit_step: int | None
    final_target_prob: float
    max_steps: int
    trajectory_steps: tuple[int, ...]
    trajectory_probs: tuple[float, ...]

    @property
    def censored(self) -> bool:
        return self.steps_to_threshold is None


def _make_optimizer(name: str, parameters: list[Tensor]) -> torch.optim.Optimizer:
    learning_rate = OPTIMIZER_LEARNING_RATES[name]
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"unknown optimizer {name!r}")


def _estimator_loss(
    estimator: str,
    logits: Tensor,
    teacher_probs: Tensor,
    *,
    batch_size: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor | None]:
    """Return (loss, sampled_actions or None) for one optimizer step."""

    log_probs = logits.log_softmax(dim=-1)
    if estimator == "fkl_exact":
        report = opd_fkl_loss(FKLBatch(student_log_probs=log_probs, teacher_probs=teacher_probs))
        return report.loss, None
    if estimator == "rkl_exact":
        probs = log_probs.exp()
        value = (probs * (log_probs - teacher_probs.log())).sum()
        return value, None
    if generator is None:
        raise ValueError(f"{estimator} requires a generator")
    if estimator == "sft_sampled":
        demo = sample_categorical(teacher_probs.log(), batch_size, generator=generator)
        report = sft_loss(SFTBatch(student_logp_demo=log_probs[demo.action]))
        return report.loss, demo.action
    if estimator == "rkl_sampled":
        rollout = sample_categorical(logits.detach(), batch_size, generator=generator)
        report = opd_rkl_loss(
            RKLBatch(
                student_logp=log_probs[rollout.action],
                behavior_logp=rollout.behavior_logp,
                teacher_logp=teacher_probs.log()[rollout.action],
            )
        )
        return report.loss, rollout.action
    raise ValueError(f"unknown estimator {estimator!r}")


def run_unlock(
    problem: UnlockProblem,
    estimator: str,
    optimizer_name: str,
    *,
    threshold: float = 0.5,
    max_steps: int = 100_000,
    batch_size: int = 64,
    seed: int | None = None,
    record_every: int = 50,
) -> UnlockRun:
    logits = torch.nn.Parameter(problem.initial_logits())
    optimizer = _make_optimizer(optimizer_name, [logits])
    teacher = problem.teacher_probs()
    generator: torch.Generator | None = None
    if estimator in SAMPLED_ESTIMATORS:
        if seed is None:
            raise ValueError("sampled estimators require a seed")
        generator = torch.Generator().manual_seed(seed)

    steps_to_threshold: int | None = None
    first_hit_step: int | None = None
    trajectory_steps: list[int] = []
    trajectory_probs: list[float] = []
    target_prob = float(logits.detach().softmax(dim=-1)[problem.target_index])
    for step in range(max_steps):
        if step % record_every == 0:
            trajectory_steps.append(step)
            trajectory_probs.append(target_prob)
        optimizer.zero_grad(set_to_none=True)
        loss, actions = _estimator_loss(
            estimator, logits, teacher, batch_size=batch_size, generator=generator
        )
        loss.backward()
        optimizer.step()
        if (
            first_hit_step is None
            and actions is not None
            and bool((actions == problem.target_index).any())
        ):
            first_hit_step = step
        target_prob = float(logits.detach().softmax(dim=-1)[problem.target_index])
        if target_prob >= threshold:
            steps_to_threshold = step + 1
            break
    trajectory_steps.append(steps_to_threshold if steps_to_threshold is not None else max_steps)
    trajectory_probs.append(target_prob)
    return UnlockRun(
        estimator=estimator,
        optimizer=optimizer_name,
        initial_target_prob=problem.initial_target_prob,
        seed=seed,
        steps_to_threshold=steps_to_threshold,
        first_hit_step=first_hit_step,
        final_target_prob=target_prob,
        max_steps=max_steps,
        trajectory_steps=tuple(trajectory_steps),
        trajectory_probs=tuple(trajectory_probs),
    )


@dataclass(frozen=True)
class UnlockStudy:
    runs: tuple[UnlockRun, ...]

    def summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        keys = sorted(
            {(run.estimator, run.optimizer, run.initial_target_prob) for run in self.runs}
        )
        for estimator, optimizer_name, p0 in keys:
            group = [
                run
                for run in self.runs
                if (run.estimator, run.optimizer, run.initial_target_prob)
                == (estimator, optimizer_name, p0)
            ]
            finished = [run.steps_to_threshold for run in group if not run.censored]
            hits = [run.first_hit_step for run in group if run.first_hit_step is not None]
            rows.append(
                {
                    "estimator": estimator,
                    "optimizer": optimizer_name,
                    "p0": p0,
                    "runs": len(group),
                    "censored": sum(run.censored for run in group),
                    "median_steps": float(np.median(finished)) if finished else math.nan,
                    "min_steps": min(finished) if finished else None,
                    "max_steps_observed": max(finished) if finished else None,
                    "median_first_hit": float(np.median(hits)) if hits else None,
                }
            )
        return rows

    def scaling_rows(self) -> list[dict[str, object]]:
        """Least-squares slope of log10(median steps) vs log10(1/p0)."""

        rows: list[dict[str, object]] = []
        summary = self.summary_rows()
        for estimator in sorted({row["estimator"] for row in summary}):
            for optimizer_name in sorted({row["optimizer"] for row in summary}):
                points = [
                    (math.log10(1.0 / float(row["p0"])), math.log10(float(row["median_steps"])))
                    for row in summary
                    if row["estimator"] == estimator
                    and row["optimizer"] == optimizer_name
                    and row["censored"] == 0
                    and not math.isnan(float(row["median_steps"]))
                ]
                if len(points) < 3:
                    slope = math.nan
                else:
                    xs, ys = zip(*points)
                    slope = float(np.polyfit(xs, ys, 1)[0])
                rows.append(
                    {
                        "estimator": estimator,
                        "optimizer": optimizer_name,
                        "fitted_points": len(points),
                        "scaling_exponent": slope,
                    }
                )
        return rows


def run_unlock_study(
    *,
    initial_target_probs: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
    optimizers: tuple[str, ...] = ("sgd", "adam"),
    sampled_seeds: tuple[int, ...] = tuple(range(8)),
    batch_size: int = 64,
    exact_max_steps: int = 100_000,
    sampled_max_steps: int = 20_000,
    threshold: float = 0.5,
) -> UnlockStudy:
    runs: list[UnlockRun] = []
    for p0 in initial_target_probs:
        problem = UnlockProblem(initial_target_prob=p0)
        for optimizer_name in optimizers:
            for estimator in EXACT_ESTIMATORS:
                runs.append(
                    run_unlock(
                        problem,
                        estimator,
                        optimizer_name,
                        threshold=threshold,
                        max_steps=exact_max_steps,
                    )
                )
            for estimator in SAMPLED_ESTIMATORS:
                for seed in sampled_seeds:
                    runs.append(
                        run_unlock(
                            problem,
                            estimator,
                            optimizer_name,
                            threshold=threshold,
                            max_steps=sampled_max_steps,
                            batch_size=batch_size,
                            seed=seed,
                        )
                    )
    return UnlockStudy(runs=tuple(runs))


def run_cold_unlock_study(
    *,
    initial_target_probs: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    optimizers: tuple[str, ...] = ("sgd", "adam"),
    sampled_seeds: tuple[int, ...] = tuple(range(8)),
    batch_size: int = 64,
    teacher_target_prob: float = 0.05,
    exact_max_steps: int = 100_000,
    sampled_max_steps: int = 60_000,
) -> UnlockStudy:
    """Cold-target variant: teacher-matched student, small-mass target.

    Threshold is 80% of the teacher's target mass.  The bulk-surplus channel
    is gone, so any fast sampled-RKL unlocking must come from either direct
    hits (rate ``~ N p_t`` per batch) or the thin uniform surplus that Adam's
    per-coordinate normalization can amplify.
    """

    threshold = 0.8 * teacher_target_prob
    runs: list[UnlockRun] = []
    for p0 in initial_target_probs:
        problem = UnlockProblem(
            teacher_target_prob=teacher_target_prob,
            initial_target_prob=p0,
            student_matched_to_teacher=True,
        )
        for optimizer_name in optimizers:
            for estimator in EXACT_ESTIMATORS:
                runs.append(
                    run_unlock(
                        problem,
                        estimator,
                        optimizer_name,
                        threshold=threshold,
                        max_steps=exact_max_steps,
                    )
                )
            for estimator in SAMPLED_ESTIMATORS:
                for seed in sampled_seeds:
                    runs.append(
                        run_unlock(
                            problem,
                            estimator,
                            optimizer_name,
                            threshold=threshold,
                            max_steps=sampled_max_steps,
                            batch_size=batch_size,
                            seed=seed,
                        )
                    )
    return UnlockStudy(runs=tuple(runs))


def exact_initial_target_gradients(problem: UnlockProblem) -> dict[str, float]:
    """Closed-form descent signal on the target logit at initialization.

    Forward KL: ``q_t - p_t`` (order one).  Reverse KL:
    ``p_t (log(q_t/p_t) + KL(p||q))`` -> order ``p_t log(1/p_t)`` as
    ``p_t -> 0``.  Used by tests to pin the mechanism behind the measured
    scaling exponents to the analytic vector fields.
    """

    teacher = problem.teacher_probs()
    logits = problem.initial_logits()
    probs = logits.softmax(dim=-1)
    target = problem.target_index
    forward = float(teacher[target] - probs[target])
    kl_value = float((probs * (probs.log() - teacher.log())).sum())
    reverse = float(probs[target] * (float(teacher.log()[target] - probs.log()[target]) + kl_value))
    return {"fkl_exact": forward, "rkl_exact": reverse}
