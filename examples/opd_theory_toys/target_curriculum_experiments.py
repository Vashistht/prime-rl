"""Teacher-target mismatch and a support-acquisition curriculum.

This is the smallest categorical problem that separates *acquisition* from
*selection*.  There are three actions: neutral, desirable, and harmful.  The
teacher and trusted utility are

``q = (0.60, 0.35, 0.05)``,       ``U = (0, +1, -1)``.

The student starts near the neutral simplex vertex,
``p = (1 - 2 eps, eps, eps)``.  Distillation can give the student substantial
desirable support, but it must also copy the teacher's harmful mass.  RL has
the correct selector objective, yet Prime-like within-group reward centering
gives exactly zero update for a homogeneous group.  For group size ``G`` and
``B`` independent groups, the exact probabilities are

``P(group informative) = 1 - sum_a p(a)**G``

``P(batch informative) = 1 - (sum_a p(a)**G)**B``.

The staged methods therefore distill only until the *pre-registered* batch
informativeness threshold is reached, then spend every remaining update on
RL.  Every trainable loss, sampler, group-advantage transform, common metric,
and KL-matched update comes from :mod:`toy_methods`; this file only defines
the problem, schedule, accounting, and exact finite-support enumeration.

Strict zero support is kept separate from epsilon support.  Ordinary finite
softmax logits always have positive support.  Moreover, once a nonzero
surrogate is observed, the softmax normalizer couples even unsampled logits.
The informative-group calculation describes the reward-contrast gate, not an
assertion that direct action sampling is the only parameter-coupling route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
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
        categorical_metrics,
        fixed_gradient_step_,
        group_advantages,
        kl_matched_step_,
        opd_fkl_loss,
        opd_rkl_loss,
        opd_rkl_population_loss,
        rl_loss,
        rl_population_loss,
        sample_categorical,
        sft_loss,
        sft_population_loss,
    )
except ImportError:  # pragma: no cover - standalone example usage
    from toy_methods import (
        FKLBatch,
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_kl,
        categorical_metrics,
        fixed_gradient_step_,
        group_advantages,
        kl_matched_step_,
        opd_fkl_loss,
        opd_rkl_loss,
        opd_rkl_population_loss,
        rl_loss,
        rl_population_loss,
        sample_categorical,
        sft_loss,
        sft_population_loss,
    )


DTYPE = torch.float64
ActionName = Literal["neutral", "desirable", "harmful"]
Phase = Literal["sft", "opd_rkl", "opd_fkl", "rl"]
Schedule = Literal["pure_rl", "sft_then_rl", "opd_rkl_then_rl", "opd_fkl_then_rl"]
StepMode = Literal["fixed_scale", "kl_matched"]
SCHEDULES: tuple[Schedule, ...] = (
    "pure_rl",
    "sft_then_rl",
    "opd_rkl_then_rl",
    "opd_fkl_then_rl",
)


def _validate_distribution(name: str, value: Tensor) -> None:
    if value.ndim != 1 or value.numel() != 3:
        raise ValueError(f"{name} must be a length-three vector")
    if not value.is_floating_point() or bool(torch.any(~torch.isfinite(value))):
        raise ValueError(f"{name} must be finite and floating point")
    if bool(torch.any(value < 0)):
        raise ValueError(f"{name} must be non-negative")
    if not torch.allclose(value.sum(), torch.ones((), dtype=value.dtype, device=value.device), atol=1e-12):
        raise ValueError(f"{name} must sum to one")


@dataclass(frozen=True)
class TargetProblem:
    """A teacher distribution paired with a separately trusted utility."""

    initial_probs: Tensor
    teacher_probs: Tensor
    utility: Tensor

    def __post_init__(self) -> None:
        _validate_distribution("initial_probs", self.initial_probs)
        _validate_distribution("teacher_probs", self.teacher_probs)
        if self.utility.shape != (3,) or not self.utility.is_floating_point():
            raise ValueError("utility must be a floating-point length-three vector")
        if bool(torch.any(~torch.isfinite(self.utility))):
            raise ValueError("utility must be finite")
        tensors = (self.initial_probs, self.teacher_probs, self.utility)
        if len({tensor.dtype for tensor in tensors}) != 1 or len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all problem tensors must share dtype and device")


def make_target_problem(
    epsilon: float = 5e-4,
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cpu",
) -> TargetProblem:
    """Build the fixed three-action experiment with finite epsilon support.

    The default ``eps=5e-4`` is fixed by a sampling-scale criterion, not by a
    favorable seed: with the default ``G=4``, ``B=8``, and ``T=60``, a policy
    frozen at initialization has about ``T * P(batch informative) = 1.9``
    informative batches in expectation.  Pure RL is therefore in an O(1)
    discovery regime where both success and failure should be visible.
    """

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between zero and one half")
    tensor = lambda values: torch.tensor(values, dtype=dtype, device=device)  # noqa: E731
    return TargetProblem(
        initial_probs=tensor((1.0 - 2.0 * epsilon, epsilon, epsilon)),
        teacher_probs=tensor((0.60, 0.35, 0.05)),
        utility=tensor((0.0, 1.0, -1.0)),
    )


class CategoricalPolicy(nn.Module):
    """Unconstrained logits initialized from a full-support distribution."""

    def __init__(self, probabilities: Tensor) -> None:
        super().__init__()
        _validate_distribution("probabilities", probabilities)
        if bool(torch.any(probabilities <= 0)):
            raise ValueError("finite-logit policy initialization requires full support")
        self.logits = nn.Parameter(probabilities.log().clone())

    @property
    def log_probs(self) -> Tensor:
        return self.logits.log_softmax(dim=-1)

    @property
    def probs(self) -> Tensor:
        return self.log_probs.exp()


def informative_group_probability(probs: Tensor, group_size: int) -> Tensor:
    """Exact probability that a group contains at least two reward values.

    Utilities are distinct for all three actions in this problem, so a group
    is uninformative exactly when every action in it is identical.
    """

    _validate_distribution("probs", probs)
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    return 1.0 - probs.pow(group_size).sum()


def informative_batch_probability(probs: Tensor, group_size: int, group_count: int) -> Tensor:
    """Exact probability that at least one of ``group_count`` groups informs."""

    if not isinstance(group_count, int) or isinstance(group_count, bool) or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    homogeneous_group = 1.0 - informative_group_probability(probs, group_size)
    return 1.0 - homogeneous_group.pow(group_count)


@dataclass(frozen=True)
class ExactGradients:
    """Population gradient-descent directions in the common logit chart."""

    probabilities: tuple[float, float, float]
    sft_fkl: tuple[float, float, float]
    opd_rkl: tuple[float, float, float]
    rl: tuple[float, float, float]
    grouped_rl: tuple[float, float, float]
    grouped_rl_scale: float


def _descent_direction(loss_value: Tensor, policy: CategoricalPolicy) -> Tensor:
    return -torch.autograd.grad(loss_value, policy.logits)[0].detach()


def exact_population_gradients(
    problem: TargetProblem,
    *,
    probabilities: Tensor | None = None,
    group_size: int = 4,
) -> ExactGradients:
    """Evaluate exact objectives from ``toy_methods`` before Monte Carlo.

    For unnormalized centered group rewards, the expected policy-gradient
    direction is ``(G - 1) / G`` times the ordinary population RL direction.
    ``enumerated_grouped_rl_gradient`` independently checks this identity by
    summing every possible sampled group through the shared RL surrogate.
    """

    probs = problem.initial_probs if probabilities is None else probabilities
    _validate_distribution("probabilities", probs)
    if bool(torch.any(probs <= 0)):
        raise ValueError("finite-logit population gradients require full support")
    if group_size < 1:
        raise ValueError("group_size must be positive")

    policy = CategoricalPolicy(probs)
    sft_direction = _descent_direction(
        sft_population_loss(policy.logits, problem.teacher_probs), policy
    )
    policy = CategoricalPolicy(probs)
    opd_direction = _descent_direction(
        opd_rkl_population_loss(policy.logits, problem.teacher_probs), policy
    )
    policy = CategoricalPolicy(probs)
    rl_direction = _descent_direction(rl_population_loss(policy.logits, problem.utility), policy)
    scale = (group_size - 1.0) / group_size
    grouped = scale * rl_direction

    to_tuple = lambda value: tuple(float(item) for item in value)  # noqa: E731
    return ExactGradients(
        probabilities=to_tuple(probs),
        sft_fkl=to_tuple(sft_direction),
        opd_rkl=to_tuple(opd_direction),
        rl=to_tuple(rl_direction),
        grouped_rl=to_tuple(grouped),
        grouped_rl_scale=scale,
    )


def enumerated_grouped_rl_gradient(
    problem: TargetProblem,
    *,
    probabilities: Tensor | None = None,
    group_size: int = 4,
) -> Tensor:
    """Enumerate the exact fresh-policy gradient of centered grouped RL.

    Sampling probabilities are detached, as they are for an on-policy policy
    gradient estimator.  Each conditional surrogate itself is evaluated by
    :func:`toy_methods.rl_loss` and uses :func:`toy_methods.group_advantages`.
    """

    probs = problem.initial_probs if probabilities is None else probabilities
    _validate_distribution("probabilities", probs)
    if bool(torch.any(probs <= 0)):
        raise ValueError("enumeration requires full support")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    policy = CategoricalPolicy(probs)
    log_probs = policy.log_probs
    expected_surrogate = torch.zeros((), dtype=probs.dtype, device=probs.device)
    for action_tuple in product(range(3), repeat=group_size):
        actions = torch.tensor(action_tuple, dtype=torch.long, device=probs.device)
        sampling_weight = probs.detach().gather(0, actions).prod()
        behavior_logp = probs.detach().log().gather(0, actions)
        student_logp = log_probs.gather(0, actions)
        rewards = problem.utility.gather(0, actions).unsqueeze(0)
        advantages = group_advantages(rewards, normalize=False).squeeze(0)
        conditional = rl_loss(RLBatch(student_logp, behavior_logp, advantages)).loss
        expected_surrogate = expected_surrogate + sampling_weight * conditional
    return torch.autograd.grad(expected_surrogate, policy.logits)[0].detach()


@dataclass(frozen=True)
class TheoryPredictions:
    """Claims fixed from objectives and sampling geometry before seeded runs."""

    initial_probs: tuple[float, float, float]
    teacher_optimum: tuple[float, float, float]
    rl_optimum: tuple[float, float, float]
    teacher_expected_utility: float
    initial_expected_utility: float
    initial_group_informative_probability: float
    initial_batch_informative_probability: float
    initial_expected_informative_batches: float
    switch_probability: float
    switch_rule: str
    support_note: str
    grouped_rl_scale: float
    initial_gradients: ExactGradients
    preregistered_predictions: tuple[str, ...]


@dataclass(frozen=True)
class NoOpCheck:
    """Empirical check of the analytic centered-reward no-op gate."""

    trials: int
    group_size: int
    group_count: int
    predicted_batch_no_op_probability: float
    observed_batch_no_op_probability: float
    standard_error: float


def measure_grouped_rl_no_op_probability(
    probs: Tensor,
    *,
    group_size: int,
    group_count: int,
    trials: int,
    seed: int,
    utility: Tensor | None = None,
) -> NoOpCheck:
    """Measure the batch no-op frequency using shared sampling/advantages."""

    _validate_distribution("probs", probs)
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if utility is None:
        utility = torch.tensor((0.0, 1.0, -1.0), dtype=probs.dtype, device=probs.device)
    if utility.shape != probs.shape or bool(torch.any(~torch.isfinite(utility))):
        raise ValueError("utility must be a finite vector matching probs")
    if torch.unique(utility).numel() != utility.numel():
        raise ValueError("the closed-form action-homogeneity check requires distinct utilities")
    sample_count = trials * group_count * group_size
    generator = torch.Generator(device=probs.device).manual_seed(seed)
    samples = sample_categorical(probs.log(), sample_count, generator=generator)
    actions = samples.action.reshape(trials, group_count, group_size)
    rewards = utility.gather(0, actions.reshape(-1)).reshape_as(actions)
    advantages = group_advantages(rewards, normalize=False)
    observed_no_op = ~(advantages != 0).any(dim=-1).any(dim=-1)
    predicted = 1.0 - informative_batch_probability(probs, group_size, group_count)
    predicted_value = float(predicted)
    standard_error = (predicted_value * (1.0 - predicted_value) / trials) ** 0.5
    return NoOpCheck(
        trials=trials,
        group_size=group_size,
        group_count=group_count,
        predicted_batch_no_op_probability=predicted_value,
        observed_batch_no_op_probability=float(observed_no_op.to(probs.dtype).mean()),
        standard_error=standard_error,
    )


@dataclass(frozen=True)
class RunConfig:
    """Budgets and the outcome-independent curriculum rule."""

    total_updates: int = 60
    acquisition_batch_size: int = 32
    group_size: int = 4
    group_count: int = 8
    step_mode: StepMode = "fixed_scale"
    fixed_step_scale: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 2e-3
    switch_probability: float = 0.90
    initial_step_scale: float = 1.0
    max_step_scale: float = 1e6

    def __post_init__(self) -> None:
        integers = (self.total_updates, self.acquisition_batch_size, self.group_size, self.group_count)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers):
            raise ValueError("update, batch, and group counts must be positive integers")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive")
        if self.step_mode not in {"fixed_scale", "kl_matched"}:
            raise ValueError(f"unknown step_mode {self.step_mode!r}")
        if self.fixed_step_scale <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("fixed_step_scale and max_grad_norm must be positive")
        if not 0.0 < self.switch_probability < 1.0:
            raise ValueError("switch_probability must lie in (0, 1)")
        if not 0.0 < self.initial_step_scale <= self.max_step_scale:
            raise ValueError("require 0 < initial_step_scale <= max_step_scale")


def preregister_predictions(
    problem: TargetProblem | None = None,
    config: RunConfig = RunConfig(),
) -> TheoryPredictions:
    """Return predictions that do not depend on any sampled trajectory."""

    problem = make_target_problem() if problem is None else problem
    gradients = exact_population_gradients(problem, group_size=config.group_size)
    initial_log_probs = problem.initial_probs.log()
    teacher_log_probs = problem.teacher_probs.log()
    initial_metrics = categorical_metrics(
        initial_log_probs,
        teacher_log_probs=teacher_log_probs,
        utility=problem.utility,
    )
    teacher_metrics = categorical_metrics(
        teacher_log_probs,
        teacher_log_probs=teacher_log_probs,
        utility=problem.utility,
    )
    group_probability = informative_group_probability(problem.initial_probs, config.group_size)
    batch_probability = informative_batch_probability(
        problem.initial_probs, config.group_size, config.group_count
    )
    return TheoryPredictions(
        initial_probs=gradients.probabilities,
        teacher_optimum=tuple(float(value) for value in problem.teacher_probs),
        rl_optimum=(0.0, 1.0, 0.0),
        teacher_expected_utility=float(teacher_metrics["expected_utility"]),
        initial_expected_utility=float(initial_metrics["expected_utility"]),
        initial_group_informative_probability=float(group_probability),
        initial_batch_informative_probability=float(batch_probability),
        initial_expected_informative_batches=config.total_updates * float(batch_probability),
        switch_probability=config.switch_probability,
        switch_rule=(
            "At the beginning of each update, switch irreversibly when the exact "
            "current-policy analytic batch-informativeness reaches the fixed threshold."
        ),
        support_note=(
            "Strict simplex zeros are unreachable with finite softmax logits; epsilon support "
            "has nonzero but asymptotically vanishing RKL/RL signal."
        ),
        grouped_rl_scale=gradients.grouped_rl_scale,
        initial_gradients=gradients,
        preregistered_predictions=(
            "SFT and both KL directions converge to q, copying useful and harmful teacher mass.",
            "Unregularized RL selects the desirable vertex and removes the teacher's harmful mass.",
            "Near the neutral vertex, distillation acquires support faster than grouped RL.",
            "Switching once analytic batch informativeness crosses the fixed threshold combines acquisition and selection.",
            "Pure RL has seed-dependent waiting time because homogeneous reward groups have exactly zero centered advantage.",
        ),
    )


@dataclass(frozen=True)
class CostVector:
    """Observable training calls; no claim that unlike calls have equal price."""

    teacher_hard_labels: int = 0
    teacher_logprob_evals: int = 0
    teacher_full_vectors: int = 0
    student_rollouts: int = 0
    reward_calls: int = 0

    def __add__(self, other: CostVector) -> CostVector:
        if not isinstance(other, CostVector):
            return NotImplemented
        return CostVector(
            teacher_hard_labels=self.teacher_hard_labels + other.teacher_hard_labels,
            teacher_logprob_evals=self.teacher_logprob_evals + other.teacher_logprob_evals,
            teacher_full_vectors=self.teacher_full_vectors + other.teacher_full_vectors,
            student_rollouts=self.student_rollouts + other.student_rollouts,
            reward_calls=self.reward_calls + other.reward_calls,
        )


@dataclass(frozen=True)
class UpdateRecord:
    schedule: Schedule
    seed: int
    update: int
    phase: Phase
    step_mode: StepMode
    neutral_probability: float
    desirable_probability: float
    harmful_probability: float
    expected_utility: float
    kl_teacher_student: float
    kl_student_teacher: float
    informative_group_probability: float
    informative_batch_probability: float
    observed_informative_groups: int | None
    raw_grad_norm: float
    applied_step_scale: float
    step_clipped: bool
    no_op: bool
    step_bracketed: bool | None
    realized_kl: float
    cumulative_cost: CostVector

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        cost = record.pop("cumulative_cost")
        record.update({f"cost_{key}": value for key, value in cost.items()})
        return record


@dataclass(frozen=True)
class RunSummary:
    schedule: Schedule
    seed: int
    step_mode: StepMode
    switch_update: int | None
    rl_updates: int
    informative_rl_updates: int
    no_op_updates: int
    final_neutral_probability: float
    final_desirable_probability: float
    final_harmful_probability: float
    final_expected_utility: float
    final_kl_teacher_student: float
    total_cost: CostVector

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        cost = record.pop("total_cost")
        record.update({f"cost_{key}": value for key, value in cost.items()})
        return record


@dataclass(frozen=True)
class CurriculumStudy:
    predictions: TheoryPredictions
    config: RunConfig
    summaries: tuple[RunSummary, ...]
    trajectories: tuple[UpdateRecord, ...]

    def summary_records(self) -> list[dict[str, object]]:
        return [summary.as_record() for summary in self.summaries]

    def trajectory_records(self) -> list[dict[str, object]]:
        return [record.as_record() for record in self.trajectories]


@dataclass(frozen=True)
class ScaleComparison:
    """Call-efficiency and conditional movement views kept intentionally apart."""

    fixed_scale: CurriculumStudy
    kl_matched: CurriculumStudy


def _phase_for_schedule(schedule: Schedule) -> Phase | None:
    mapping: dict[Schedule, Phase | None] = {
        "pure_rl": None,
        "sft_then_rl": "sft",
        "opd_rkl_then_rl": "opd_rkl",
        "opd_fkl_then_rl": "opd_fkl",
    }
    try:
        return mapping[schedule]
    except KeyError as error:
        raise ValueError(f"unknown schedule {schedule!r}") from error


def _acquisition_loss_and_cost(
    policy: CategoricalPolicy,
    problem: TargetProblem,
    phase: Literal["sft", "opd_rkl", "opd_fkl"],
    config: RunConfig,
    generator: torch.Generator,
) -> tuple[Tensor, CostVector]:
    n = config.acquisition_batch_size
    if phase == "sft":
        demonstrations = sample_categorical(problem.teacher_probs.log(), n, generator=generator)
        student_logp = policy.log_probs.gather(0, demonstrations.action)
        return sft_loss(SFTBatch(student_logp)).loss, CostVector(teacher_hard_labels=n)
    if phase == "opd_rkl":
        samples = sample_categorical(policy.logits, n, generator=generator)
        student_logp = policy.log_probs.gather(0, samples.action)
        teacher_logp = problem.teacher_probs.log().gather(0, samples.action)
        batch = RKLBatch(student_logp, samples.behavior_logp, teacher_logp)
        return opd_rkl_loss(batch).loss, CostVector(
            teacher_logprob_evals=n,
            student_rollouts=n,
        )
    if phase == "opd_fkl":
        batch = FKLBatch(policy.log_probs.unsqueeze(0), problem.teacher_probs.unsqueeze(0))
        return opd_fkl_loss(batch).loss, CostVector(
            teacher_logprob_evals=problem.teacher_probs.numel(),
            teacher_full_vectors=1,
        )
    raise ValueError(f"unknown acquisition phase {phase!r}")


def _rl_loss_cost_and_groups(
    policy: CategoricalPolicy,
    problem: TargetProblem,
    config: RunConfig,
    generator: torch.Generator,
) -> tuple[Tensor, CostVector, int]:
    sample_count = config.group_size * config.group_count
    samples = sample_categorical(policy.logits, sample_count, generator=generator)
    actions = samples.action.reshape(config.group_count, config.group_size)
    behavior_logp = samples.behavior_logp.reshape(config.group_count, config.group_size)
    student_logp = policy.log_probs.gather(0, samples.action).reshape_as(actions)
    rewards = problem.utility.gather(0, actions.reshape(-1)).reshape_as(actions)
    advantages = group_advantages(rewards, normalize=False)
    report = rl_loss(
        RLBatch(
            student_logp=student_logp.reshape(-1),
            behavior_logp=behavior_logp.reshape(-1),
            advantage=advantages.reshape(-1),
        )
    )
    observed = int((actions != actions[:, :1]).any(dim=-1).sum())
    cost = CostVector(student_rollouts=sample_count, reward_calls=sample_count)
    return report.loss, cost, observed


def _record(
    *,
    schedule: Schedule,
    seed: int,
    update: int,
    phase: Phase,
    policy: CategoricalPolicy,
    problem: TargetProblem,
    config: RunConfig,
    observed_informative_groups: int | None,
    raw_grad_norm: float,
    applied_step_scale: float,
    step_clipped: bool,
    no_op: bool,
    step_bracketed: bool | None,
    realized_kl: float,
    cumulative_cost: CostVector,
) -> UpdateRecord:
    log_probs = policy.log_probs.detach()
    probs = log_probs.exp()
    metrics = categorical_metrics(
        log_probs,
        teacher_log_probs=problem.teacher_probs.log(),
        utility=problem.utility,
    )
    return UpdateRecord(
        schedule=schedule,
        seed=seed,
        update=update,
        phase=phase,
        step_mode=config.step_mode,
        neutral_probability=float(probs[0]),
        desirable_probability=float(probs[1]),
        harmful_probability=float(probs[2]),
        expected_utility=float(metrics["expected_utility"]),
        kl_teacher_student=float(metrics["kl_teacher_student"]),
        kl_student_teacher=float(metrics["kl_student_teacher"]),
        informative_group_probability=float(informative_group_probability(probs, config.group_size)),
        informative_batch_probability=float(
            informative_batch_probability(probs, config.group_size, config.group_count)
        ),
        observed_informative_groups=observed_informative_groups,
        raw_grad_norm=raw_grad_norm,
        applied_step_scale=applied_step_scale,
        step_clipped=step_clipped,
        no_op=no_op,
        step_bracketed=step_bracketed,
        realized_kl=realized_kl,
        cumulative_cost=cumulative_cost,
    )


def run_schedule(
    schedule: Schedule,
    *,
    seed: int,
    problem: TargetProblem | None = None,
    config: RunConfig = RunConfig(),
) -> tuple[RunSummary, tuple[UpdateRecord, ...]]:
    """Run one fixed-budget schedule with per-nonzero-update KL matching."""

    problem = make_target_problem() if problem is None else problem
    acquisition_phase = _phase_for_schedule(schedule)
    policy = CategoricalPolicy(problem.initial_probs)
    generator = torch.Generator(device=problem.initial_probs.device).manual_seed(seed)
    cost = CostVector()
    switch_update: int | None = 0 if schedule == "pure_rl" else None
    switched = schedule == "pure_rl"
    records: list[UpdateRecord] = []
    rl_updates = 0
    informative_rl_updates = 0
    no_op_updates = 0

    for update in range(config.total_updates):
        batch_informative = informative_batch_probability(
            policy.probs.detach(), config.group_size, config.group_count
        )
        if acquisition_phase is None or switched or float(batch_informative) >= config.switch_probability:
            phase: Phase = "rl"
            if switch_update is None:
                switch_update = update
                switched = True
            loss_value, update_cost, observed_groups = _rl_loss_cost_and_groups(
                policy, problem, config, generator
            )
            rl_updates += 1
            informative_rl_updates += int(observed_groups > 0)
        else:
            phase = acquisition_phase
            loss_value, update_cost = _acquisition_loss_and_cost(
                policy, problem, phase, config, generator
            )
            observed_groups = None

        old_probs = policy.probs.detach().clone()

        def behavior_kl() -> Tensor:
            return categorical_kl(old_probs, policy.probs)

        if config.step_mode == "fixed_scale":
            step = fixed_gradient_step_(
                policy,
                loss_value,
                behavior_kl,
                step_scale=config.fixed_step_scale,
                max_grad_norm=config.max_grad_norm,
            )
            applied_step_scale = step.applied_scale
            step_clipped = step.clipped
            no_op = step.no_op
            step_bracketed = None
        else:
            matched_step = kl_matched_step_(
                policy,
                loss_value,
                behavior_kl,
                target_kl=config.target_kl,
                initial_scale=config.initial_step_scale,
                max_scale=config.max_step_scale,
            )
            step = matched_step
            applied_step_scale = matched_step.scale
            step_clipped = False
            no_op = matched_step.grad_norm == 0.0
            step_bracketed = matched_step.bracketed
        no_op_updates += int(no_op)
        cost = cost + update_cost
        records.append(
            _record(
                schedule=schedule,
                seed=seed,
                update=update,
                phase=phase,
                policy=policy,
                problem=problem,
                config=config,
                observed_informative_groups=observed_groups,
                raw_grad_norm=step.grad_norm,
                applied_step_scale=applied_step_scale,
                step_clipped=step_clipped,
                no_op=no_op,
                step_bracketed=step_bracketed,
                realized_kl=step.realized_kl,
                cumulative_cost=cost,
            )
        )

    final = records[-1]
    summary = RunSummary(
        schedule=schedule,
        seed=seed,
        step_mode=config.step_mode,
        switch_update=switch_update,
        rl_updates=rl_updates,
        informative_rl_updates=informative_rl_updates,
        no_op_updates=no_op_updates,
        final_neutral_probability=final.neutral_probability,
        final_desirable_probability=final.desirable_probability,
        final_harmful_probability=final.harmful_probability,
        final_expected_utility=final.expected_utility,
        final_kl_teacher_student=final.kl_teacher_student,
        total_cost=cost,
    )
    return summary, tuple(records)


def run_curriculum_study(
    *,
    seeds: Sequence[int] = tuple(range(8)),
    schedules: Sequence[Schedule] = SCHEDULES,
    problem: TargetProblem | None = None,
    config: RunConfig = RunConfig(),
) -> CurriculumStudy:
    """Run seeded schedules after constructing the independent predictions."""

    if not seeds:
        raise ValueError("seeds must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not schedules:
        raise ValueError("schedules must be non-empty")
    if len(set(schedules)) != len(schedules):
        raise ValueError("schedules must be unique")
    problem = make_target_problem() if problem is None else problem
    predictions = preregister_predictions(problem, config)
    summaries: list[RunSummary] = []
    trajectories: list[UpdateRecord] = []
    for schedule in schedules:
        for seed in seeds:
            summary, records = run_schedule(
                schedule,
                seed=seed,
                problem=problem,
                config=config,
            )
            summaries.append(summary)
            trajectories.extend(records)
    return CurriculumStudy(
        predictions=predictions,
        config=config,
        summaries=tuple(summaries),
        trajectories=tuple(trajectories),
    )


def run_scale_comparison(
    *,
    seeds: Sequence[int] = tuple(range(8)),
    schedules: Sequence[Schedule] = SCHEDULES,
    problem: TargetProblem | None = None,
    config: RunConfig = RunConfig(),
) -> ScaleComparison:
    """Run two labeled views without conflating their conclusions.

    ``fixed_scale`` preserves raw signal magnitude and therefore measures
    progress per fixed data-call budget.  ``kl_matched`` asks where a useful
    nonzero direction moves the policy at equal behavioral KL; it can require
    a huge applied scale for a vanishing gradient and is not evidence of equal
    call efficiency.  Both views retain raw norms, scales, KL, and no-op rates.
    """

    fixed_config = replace(config, step_mode="fixed_scale")
    matched_config = replace(config, step_mode="kl_matched")
    fixed = run_curriculum_study(
        seeds=seeds,
        schedules=schedules,
        problem=problem,
        config=fixed_config,
    )
    matched = run_curriculum_study(
        seeds=seeds,
        schedules=schedules,
        problem=problem,
        config=matched_config,
    )
    return ScaleComparison(fixed_scale=fixed, kl_matched=matched)
