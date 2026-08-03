"""A horizon-two state-coverage experiment with no sequential confounds.

The first action selects one of two terminal decision states.  Its teacher and
student distributions are frozen, so ``d_q != d_p`` is an experimental input,
not something a second-stage update can change.  The trainable policy is just
one independent categorical row per second-stage state.

This exposes three facts before any Monte Carlo run:

* a teacher-state collector cannot train a student-only state;
* on every covered tabular row, hard SFT, conditional FKL, and conditional RKL
  share the same population optimum (the teacher conditional);
* their finite-sample noise and initial update geometry still differ.

All optimization losses, samplers, reductions, policy-gradient advantages,
and KL-matched steps are imported from :mod:`toy_methods`.  This module only
defines the fork, data collection, factorial sweeps, and compact records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

try:  # Support package imports and direct execution from this directory.
    from .toy_methods import (
        FKLBatch,
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_kl,
        group_advantages,
        kl_matched_step_,
        opd_fkl_loss,
        opd_rkl_loss,
        opd_rkl_population_values,
        rl_loss,
        rl_population_values,
        sample_categorical,
        sft_loss,
        sft_population_values,
        weighted_mean,
    )
except ImportError:  # pragma: no cover - direct script use
    from toy_methods import (
        FKLBatch,
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_kl,
        group_advantages,
        kl_matched_step_,
        opd_fkl_loss,
        opd_rkl_loss,
        opd_rkl_population_values,
        rl_loss,
        rl_population_values,
        sample_categorical,
        sft_loss,
        sft_population_values,
        weighted_mean,
    )


DTYPE = torch.float64
Collector = Literal["teacher", "student"]
DistillationMethod = Literal["sft_hard", "opd_fkl", "opd_rkl"]
COLLECTORS: tuple[Collector, ...] = ("teacher", "student")
DISTILLATION_METHODS: tuple[DistillationMethod, ...] = ("sft_hard", "opd_fkl", "opd_rkl")


def _validate_distribution(name: str, value: Tensor, *, ndim: int) -> None:
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not value.is_floating_point() or bool(torch.any(~torch.isfinite(value))):
        raise ValueError(f"{name} must be a finite floating-point tensor")
    if bool(torch.any(value < 0)):
        raise ValueError(f"{name} must be non-negative")
    if not torch.allclose(value.sum(dim=-1), torch.ones_like(value.sum(dim=-1)), atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must sum to one on its final dimension")


@dataclass(frozen=True)
class TwoStepFork:
    """Frozen first-step occupancies and tabular second-step quantities."""

    teacher_occupancy: Tensor
    student_occupancy: Tensor
    initial_conditionals: Tensor
    teacher_conditionals: Tensor
    trusted_utility: Tensor
    teacher_reliability: float
    recovery_state: int = 1
    trusted_action: int = 1

    def __post_init__(self) -> None:
        _validate_distribution("teacher_occupancy", self.teacher_occupancy, ndim=1)
        _validate_distribution("student_occupancy", self.student_occupancy, ndim=1)
        _validate_distribution("initial_conditionals", self.initial_conditionals, ndim=2)
        _validate_distribution("teacher_conditionals", self.teacher_conditionals, ndim=2)
        if self.teacher_occupancy.shape != self.student_occupancy.shape:
            raise ValueError("teacher and student occupancies must have the same shape")
        expected = (self.teacher_occupancy.numel(), self.initial_conditionals.shape[-1])
        if self.initial_conditionals.shape != expected or self.teacher_conditionals.shape != expected:
            raise ValueError(f"conditional policies must have shape {expected}")
        if self.trusted_utility.shape != expected:
            raise ValueError(f"trusted_utility must have shape {expected}")
        tensors = (
            self.teacher_occupancy,
            self.student_occupancy,
            self.initial_conditionals,
            self.teacher_conditionals,
            self.trusted_utility,
        )
        if len({tensor.device for tensor in tensors}) != 1 or len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all fork tensors must share dtype and device")
        if bool(torch.any(~torch.isfinite(self.trusted_utility))):
            raise ValueError("trusted_utility must be finite")
        if not 0.0 <= self.teacher_reliability <= 1.0:
            raise ValueError("teacher_reliability must lie in [0, 1]")
        if not 0 <= self.recovery_state < expected[0]:
            raise ValueError("recovery_state is out of range")
        if not 0 <= self.trusted_action < expected[1]:
            raise ValueError("trusted_action is out of range")

    def occupancy(self, collector: Collector) -> Tensor:
        if collector == "teacher":
            return self.teacher_occupancy
        if collector == "student":
            return self.student_occupancy
        raise ValueError(f"unknown collector {collector!r}")


def make_two_step_fork(
    teacher_reliability: float,
    *,
    teacher_recovery_mass: float = 0.0,
    student_recovery_mass: float = 0.8,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> TwoStepFork:
    """Build the minimal fork used in the sweep.

    Action 1 is trusted.  On the recovery state the teacher interpolates
    between a confidently wrong conditional and a confidently correct one.
    Both endpoints remove mass from the student's distractor action 2, so a
    query is corrective even when its reallocation conflicts with reward.
    """

    if not 0.0 <= teacher_reliability <= 1.0:
        raise ValueError("teacher_reliability must lie in [0, 1]")
    if not 0.0 <= teacher_recovery_mass < 1.0:
        raise ValueError("teacher_recovery_mass must lie in [0, 1)")
    if not 0.0 < student_recovery_mass < 1.0:
        raise ValueError("student_recovery_mass must lie in (0, 1)")
    tensor = lambda values: torch.tensor(values, dtype=dtype, device=device)  # noqa: E731
    reliable = tensor([0.05, 0.90, 0.05])
    unreliable = tensor([0.90, 0.05, 0.05])
    recovery_teacher = teacher_reliability * reliable + (1.0 - teacher_reliability) * unreliable
    return TwoStepFork(
        teacher_occupancy=tensor([1.0 - teacher_recovery_mass, teacher_recovery_mass]),
        student_occupancy=tensor([1.0 - student_recovery_mass, student_recovery_mass]),
        initial_conditionals=tensor([[0.75, 0.15, 0.10], [0.10, 0.20, 0.70]]),
        teacher_conditionals=torch.stack((reliable, recovery_teacher)),
        trusted_utility=tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        teacher_reliability=float(teacher_reliability),
    )


class TabularSecondStepPolicy(nn.Module):
    """Independent logits per terminal decision state; no shared parameters."""

    def __init__(self, probabilities: Tensor) -> None:
        super().__init__()
        _validate_distribution("probabilities", probabilities, ndim=2)
        if bool(torch.any(probabilities <= 0)):
            raise ValueError("initial probabilities must have full support")
        self.logits = nn.Parameter(probabilities.log().clone())

    @property
    def log_probs(self) -> Tensor:
        return self.logits.log_softmax(dim=-1)

    @property
    def probs(self) -> Tensor:
        return self.logits.softmax(dim=-1)


def _population_values(
    method: DistillationMethod,
    policy: TabularSecondStepPolicy,
    teacher: Tensor,
) -> Tensor:
    if method == "sft_hard":
        return sft_population_values(policy.logits, teacher)
    if method == "opd_fkl":
        # The shared FKL operator reduces rows, so evaluate one row at a time
        # before applying the separate state-occupancy reduction.
        return torch.stack(
            [opd_fkl_loss(FKLBatch(policy.log_probs[state], teacher[state])).loss for state in range(teacher.shape[0])]
        )
    if method == "opd_rkl":
        return opd_rkl_population_values(policy.logits, teacher)
    raise ValueError(f"unknown method {method!r}")


def population_loss(
    method: DistillationMethod,
    policy: TabularSecondStepPolicy,
    problem: TwoStepFork,
    collector: Collector,
) -> Tensor:
    """Exact conditional objective under a frozen state collector."""

    return weighted_mean(_population_values(method, policy, problem.teacher_conditionals), problem.occupancy(collector))


def expected_trusted_utility(logits: Tensor, problem: TwoStepFork) -> Tensor:
    """Reward under the frozen student first-step occupancy."""

    utility_by_state = -rl_population_values(logits, problem.trusted_utility)
    return weighted_mean(utility_by_state, problem.student_occupancy)


@dataclass(frozen=True)
class Predictions:
    population_target: str
    coverage_prediction: str
    finite_variance_prediction: str
    reliability_prediction: str
    initial_update_prediction: str
    hard_label_information: str
    fkl_information: str
    rkl_information: str
    top_action_agreement_threshold: float
    recovery_improvement_threshold: float
    initial_recovery_trusted_probability: float


def predictions(problem: TwoStepFork | None = None) -> Predictions:
    problem = make_two_step_fork(1.0) if problem is None else problem
    initial = float(problem.initial_conditionals[problem.recovery_state, problem.trusted_action])
    wrong = 0.05
    correct = 0.90
    return Predictions(
        population_target="On every state with positive collector mass, hard SFT, FKL, and RKL all target q(a|s).",
        coverage_prediction="A zero-mass recovery state has exactly zero update and remains at its initialization.",
        finite_variance_prediction=(
            "FKL has only state-count noise; hard SFT also samples teacher labels; RKL also samples student actions."
        ),
        reliability_prediction=(
            "At the population endpoint on the recovery state, distillation improves trusted reward exactly when "
            "teacher correct-action mass exceeds the student's initial correct-action mass; one-step signs can differ."
        ),
        initial_update_prediction=(
            "For this three-action fork, every method initially raises trusted-action mass even at reliability zero, "
            "because all teachers remove the larger distractor mass; an unreliable endpoint later reverses that gain."
        ),
        hard_label_information="One teacher-sampled action label per queried state.",
        fkl_information="The complete teacher probability row q(a|s) per queried state.",
        rkl_information="A student-sampled action plus the teacher log-probability only at that action.",
        top_action_agreement_threshold=0.5,
        recovery_improvement_threshold=(initial - wrong) / (correct - wrong),
        initial_recovery_trusted_probability=initial,
    )


@dataclass(frozen=True)
class ExactRecord:
    reliability: float
    collector: Collector
    method: DistillationMethod
    identified_states: tuple[bool, bool]
    population_loss: float
    common_update: tuple[float, ...]
    recovery_update: tuple[float, ...]
    canonical_recovery_probs: tuple[float, ...]
    matched_recovery_probs: tuple[float, ...]
    matched_expected_utility: float
    matched_kl: float
    matched_step_bracketed: bool


def exact_population_records(
    reliabilities: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    *,
    target_kl: float = 0.01,
    teacher_recovery_mass: float = 0.0,
) -> list[ExactRecord]:
    """Population optima and initial KL-matched update directions, first."""

    records: list[ExactRecord] = []
    for reliability in reliabilities:
        problem = make_two_step_fork(reliability, teacher_recovery_mass=teacher_recovery_mass)
        for collector in COLLECTORS:
            occupancy = problem.occupancy(collector)
            identified = tuple(bool(value > 0) for value in occupancy)
            for method in DISTILLATION_METHODS:
                policy = TabularSecondStepPolicy(problem.initial_conditionals)
                loss_value = population_loss(method, policy, problem, collector)
                update = -torch.autograd.grad(loss_value, policy.logits, retain_graph=True)[0].detach()
                canonical = torch.where(
                    (occupancy > 0)[:, None],
                    problem.teacher_conditionals,
                    problem.initial_conditionals,
                )
                old_probs = policy.probs.detach().clone()

                def evaluation_kl() -> Tensor:
                    row_kl = categorical_kl(old_probs, policy.probs)
                    return weighted_mean(row_kl, problem.student_occupancy)

                step = kl_matched_step_(
                    policy,
                    loss_value,
                    evaluation_kl,
                    target_kl=target_kl,
                    initial_scale=1.0,
                    max_scale=1e4,
                    rtol=5e-3,
                    max_iter=24,
                )
                records.append(
                    ExactRecord(
                        reliability=float(reliability),
                        collector=collector,
                        method=method,
                        identified_states=identified,  # type: ignore[arg-type]
                        population_loss=float(loss_value.detach()),
                        common_update=tuple(float(value) for value in update[0]),
                        recovery_update=tuple(float(value) for value in update[problem.recovery_state]),
                        canonical_recovery_probs=tuple(float(value) for value in canonical[problem.recovery_state]),
                        matched_recovery_probs=tuple(float(value) for value in policy.probs[problem.recovery_state].detach()),
                        matched_expected_utility=float(expected_trusted_utility(policy.logits, problem).detach()),
                        matched_kl=step.realized_kl,
                        matched_step_bracketed=step.bracketed,
                    )
                )
    return records


def _sample_states(problem: TwoStepFork, collector: Collector, n: int, generator: torch.Generator) -> Tensor:
    return sample_categorical(problem.occupancy(collector).log(), n, generator=generator).action


def _finite_loss(
    method: DistillationMethod,
    policy: TabularSecondStepPolicy,
    states: Tensor,
    problem: TwoStepFork,
    generator: torch.Generator,
) -> Tensor:
    student_rows = policy.log_probs[states]
    teacher_rows = problem.teacher_conditionals[states]
    if method == "opd_fkl":
        return opd_fkl_loss(FKLBatch(student_rows, teacher_rows)).loss
    if method == "sft_hard":
        demonstrations = sample_categorical(teacher_rows.log(), 1, generator=generator).action.squeeze(-1)
        return sft_loss(SFTBatch(student_rows.gather(-1, demonstrations[:, None]).squeeze(-1))).loss
    if method == "opd_rkl":
        samples = sample_categorical(policy.logits[states], 1, generator=generator)
        actions = samples.action.squeeze(-1)
        return opd_rkl_loss(
            RKLBatch(
                student_logp=student_rows.gather(-1, actions[:, None]).squeeze(-1),
                behavior_logp=samples.behavior_logp.squeeze(-1),
                teacher_logp=teacher_rows.log().gather(-1, actions[:, None]).squeeze(-1),
            )
        ).loss
    raise ValueError(f"unknown method {method!r}")


@dataclass(frozen=True)
class FiniteRecord:
    reliability: float
    collector: Collector
    method: DistillationMethod
    query_count: int
    seed: int
    recovery_queries: int
    recovery_trusted_probability: float
    expected_trusted_utility: float
    realized_kl: float
    step_bracketed: bool


def finite_sample_records(
    reliabilities: Sequence[float] = (0.0, 0.5, 1.0),
    *,
    query_counts: Sequence[int] = (4, 16, 64),
    seeds: Sequence[int] = tuple(range(16)),
    target_kl: float = 0.01,
    teacher_recovery_mass: float = 0.0,
) -> list[FiniteRecord]:
    """One fresh, KL-matched update per finite query batch and seed."""

    if any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in query_counts):
        raise ValueError("query_counts must contain positive integers")
    records: list[FiniteRecord] = []
    for reliability in reliabilities:
        problem = make_two_step_fork(reliability, teacher_recovery_mass=teacher_recovery_mass)
        for collector in COLLECTORS:
            for method in DISTILLATION_METHODS:
                for query_count in query_counts:
                    for seed in seeds:
                        generator = torch.Generator(device=problem.teacher_occupancy.device).manual_seed(int(seed))
                        states = _sample_states(problem, collector, query_count, generator)
                        policy = TabularSecondStepPolicy(problem.initial_conditionals)
                        loss_value = _finite_loss(method, policy, states, problem, generator)
                        old_probs = policy.probs.detach().clone()

                        def evaluation_kl() -> Tensor:
                            return weighted_mean(
                                categorical_kl(old_probs, policy.probs),
                                problem.student_occupancy,
                            )

                        step = kl_matched_step_(
                            policy,
                            loss_value,
                            evaluation_kl,
                            target_kl=target_kl,
                            initial_scale=1.0,
                            max_scale=1e4,
                            rtol=5e-3,
                            max_iter=24,
                        )
                        records.append(
                            FiniteRecord(
                                reliability=float(reliability),
                                collector=collector,
                                method=method,
                                query_count=query_count,
                                seed=int(seed),
                                recovery_queries=int((states == problem.recovery_state).sum()),
                                recovery_trusted_probability=float(
                                    policy.probs[problem.recovery_state, problem.trusted_action].detach()
                                ),
                                expected_trusted_utility=float(expected_trusted_utility(policy.logits, problem).detach()),
                                realized_kl=step.realized_kl,
                                step_bracketed=step.bracketed,
                            )
                        )
    return records


@dataclass(frozen=True)
class FiniteSummary:
    reliability: float
    collector: Collector
    method: DistillationMethod
    query_count: int
    seeds: int
    recovery_seen_rate: float
    update_rate: float
    mean_recovery_trusted_probability: float
    std_recovery_trusted_probability: float
    mean_expected_trusted_utility: float


def summarize_finite_records(records: Sequence[FiniteRecord]) -> list[FiniteSummary]:
    """Aggregate seed records without hiding coverage or failed updates."""

    groups: dict[tuple[float, Collector, DistillationMethod, int], list[FiniteRecord]] = {}
    for record in records:
        key = (record.reliability, record.collector, record.method, record.query_count)
        groups.setdefault(key, []).append(record)
    summaries: list[FiniteSummary] = []
    for key in sorted(groups):
        group = groups[key]
        recovery = torch.tensor([record.recovery_trusted_probability for record in group], dtype=DTYPE)
        utility = torch.tensor([record.expected_trusted_utility for record in group], dtype=DTYPE)
        summaries.append(
            FiniteSummary(
                reliability=key[0],
                collector=key[1],
                method=key[2],
                query_count=key[3],
                seeds=len(group),
                recovery_seen_rate=sum(record.recovery_queries > 0 for record in group) / len(group),
                update_rate=sum(record.step_bracketed for record in group) / len(group),
                mean_recovery_trusted_probability=float(recovery.mean()),
                std_recovery_trusted_probability=float(recovery.std(correction=0)),
                mean_expected_trusted_utility=float(utility.mean()),
            )
        )
    return summaries


@dataclass(frozen=True)
class SparseRLPrediction:
    initial_jackpot_probability: float
    group_size: int
    group_informative_probability: float
    batch_informative_probability: float


@dataclass(frozen=True)
class SparseRLRecord:
    group_count: int
    group_size: int
    seed: int
    jackpot_samples: int
    informative_groups: int
    predicted_batch_informative_probability: float
    jackpot_probability_after_update: float
    realized_kl: float
    step_bracketed: bool


def sparse_rl_prediction(initial_jackpot_probability: float, group_size: int, group_count: int) -> SparseRLPrediction:
    """Probability that centered binary rewards contain any learning signal."""

    if not 0.0 < initial_jackpot_probability < 1.0:
        raise ValueError("initial_jackpot_probability must lie in (0, 1)")
    if group_size < 2 or group_count < 1:
        raise ValueError("require group_size >= 2 and group_count >= 1")
    homogeneous = (1.0 - initial_jackpot_probability) ** group_size + initial_jackpot_probability**group_size
    group_informative = 1.0 - homogeneous
    return SparseRLPrediction(
        initial_jackpot_probability=initial_jackpot_probability,
        group_size=group_size,
        group_informative_probability=group_informative,
        batch_informative_probability=1.0 - homogeneous**group_count,
    )


def sparse_action_rl_records(
    *,
    action_count: int = 16,
    initial_jackpot_probability: float = 0.01,
    group_counts: Sequence[int] = (1, 4, 16, 64),
    group_size: int = 8,
    seeds: Sequence[int] = tuple(range(64)),
    target_kl: float = 0.01,
) -> list[SparseRLRecord]:
    """Trusted-reward control on the student-only recovery branch.

    The state is deliberately fixed to the recovery row here.  This removes
    state discovery from the control: a zero update can only be caused by
    sparse action/reward discovery and group centering.
    """

    if action_count < 3:
        raise ValueError("action_count must be at least three")
    if not 0.0 < initial_jackpot_probability < 1.0:
        raise ValueError("initial_jackpot_probability must lie in (0, 1)")
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in group_counts):
        raise ValueError("group_counts must contain positive integers")
    non_jackpot = (1.0 - initial_jackpot_probability) / (action_count - 1)
    initial = torch.full((action_count,), non_jackpot, dtype=DTYPE)
    initial[0] = initial_jackpot_probability
    utility = torch.zeros(action_count, dtype=DTYPE)
    utility[0] = 1.0
    records: list[SparseRLRecord] = []
    for group_count in group_counts:
        prediction = sparse_rl_prediction(initial_jackpot_probability, group_size, group_count)
        for seed in seeds:
            generator = torch.Generator().manual_seed(int(seed))
            policy = TabularSecondStepPolicy(initial[None, :])
            samples = sample_categorical(policy.logits.expand(group_count, -1), group_size, generator=generator)
            rewards = utility[samples.action]
            advantages = group_advantages(rewards)
            student_logp = policy.log_probs.expand(group_count, -1).gather(-1, samples.action)
            loss_value = rl_loss(
                RLBatch(
                    student_logp=student_logp,
                    behavior_logp=samples.behavior_logp,
                    advantage=advantages,
                )
            ).loss
            old_probs = policy.probs.detach().clone()

            def evaluation_kl() -> Tensor:
                return weighted_mean(
                    categorical_kl(old_probs, policy.probs),
                    torch.ones(1, dtype=DTYPE),
                )

            step = kl_matched_step_(
                policy,
                loss_value,
                evaluation_kl,
                target_kl=target_kl,
                initial_scale=1.0,
                max_scale=1e6,
                rtol=5e-3,
                max_iter=24,
            )
            informative = ((rewards.sum(dim=-1) > 0) & (rewards.sum(dim=-1) < group_size)).sum()
            records.append(
                SparseRLRecord(
                    group_count=group_count,
                    group_size=group_size,
                    seed=int(seed),
                    jackpot_samples=int(rewards.sum()),
                    informative_groups=int(informative),
                    predicted_batch_informative_probability=prediction.batch_informative_probability,
                    jackpot_probability_after_update=float(policy.probs[0, 0].detach()),
                    realized_kl=step.realized_kl,
                    step_bracketed=step.bracketed,
                )
            )
    return records


@dataclass(frozen=True)
class TwoStepStudy:
    predictions: Predictions
    exact: list[ExactRecord]
    finite: list[FiniteSummary]
    sparse_rl: list[SparseRLRecord]


def run_two_step_study(
    *,
    reliabilities: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    query_counts: Sequence[int] = (4, 16, 64),
    finite_seeds: Sequence[int] = tuple(range(16)),
    sparse_seeds: Sequence[int] = tuple(range(64)),
) -> TwoStepStudy:
    """Run theory records first, then finite estimates and the RL control."""

    exact = exact_population_records(reliabilities)
    finite_raw = finite_sample_records(
        reliabilities,
        query_counts=query_counts,
        seeds=finite_seeds,
    )
    return TwoStepStudy(
        predictions=predictions(),
        exact=exact,
        finite=summarize_finite_records(finite_raw),
        sparse_rl=sparse_action_rl_records(seeds=sparse_seeds),
    )


__all__ = [
    "COLLECTORS",
    "DISTILLATION_METHODS",
    "ExactRecord",
    "FiniteRecord",
    "FiniteSummary",
    "Predictions",
    "SparseRLPrediction",
    "SparseRLRecord",
    "TabularSecondStepPolicy",
    "TwoStepFork",
    "TwoStepStudy",
    "exact_population_records",
    "expected_trusted_utility",
    "finite_sample_records",
    "make_two_step_fork",
    "population_loss",
    "predictions",
    "run_two_step_study",
    "sparse_action_rl_records",
    "sparse_rl_prediction",
    "summarize_finite_records",
]
