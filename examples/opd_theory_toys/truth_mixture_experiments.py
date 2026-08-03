"""Truth--teacher--student trajectories on a decaying named-mode mixture.

The core benchmark is intentionally equation-sized: each well-separated mode
is one categorical action.  Truth ``p_star`` has geometrically decaying weights,
teacher ``q`` may distort or omit named modes, and student ``p`` has trainable
logits.  Every objective is a direct call to the shared categorical operators
in :mod:`toy_methods`.

``sft``
    Population hard-label SFT, ``E_q[-log p]``.
``opd_fkl``
    Full-information forward KL up to fixed teacher entropy.  Its implemented
    cross-entropy scalar and gradient are identical to population SFT here.
``opd_rkl``
    Reverse KL, ``KL(p || q)``.
``truth_rl``
    Pure trusted-reward RL with ``U(a)=log p_star(a)``; it selects the largest
    reachable truth mode.
``truth_rl_entropy``
    ``E_p[log p_star] + alpha H(p)``.  At ``alpha=1`` the minimized loss is
    ``KL(p || p_star)`` on the reachable action set.

An effectively unimodal initialization gives every reachable mode a small
epsilon mass and remains recoverable.  A hard capacity mask removes logits
structurally and is not discoverable by any method.  Exact teacher omission is
also kept distinct from an epsilon-floored omission: with a full-support
student, ``KL(p || q)`` is infinite when ``q(a)=0``.  The trajectory runner
raises a labelled error instead of silently smoothing it.

Gaussian kernels render the named modes.  The core weight study is invariant
to their locations.  A separate optional *labeled-Gaussian* objective adds the
exact equal-variance location term: teacher mass weights SFT/FKL shifts, while
current student mass weights RKL and truth-RL shifts.  This gives translation
a scientifically meaningful gradient without replacing the simple core with
an unlabeled continuous-mixture optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

try:  # Support package imports and direct execution from this directory.
    from .families import DEFAULT_DTYPE, normal_log_prob
    from .toy_methods import (
        FKLBatch,
        categorical_metrics,
        opd_fkl_loss,
        opd_rkl_population_loss,
        rl_population_loss,
        sft_population_loss,
    )
except ImportError:  # pragma: no cover - standalone example usage
    from families import DEFAULT_DTYPE, normal_log_prob
    from toy_methods import (
        FKLBatch,
        categorical_metrics,
        opd_fkl_loss,
        opd_rkl_population_loss,
        rl_population_loss,
        sft_population_loss,
    )


Method = Literal["sft", "opd_fkl", "opd_rkl", "truth_rl", "truth_rl_entropy"]
DistributionName = Literal["truth", "teacher", "student"]
METHODS: tuple[Method, ...] = (
    "sft",
    "opd_fkl",
    "opd_rkl",
    "truth_rl",
    "truth_rl_entropy",
)


def _normalized_nonnegative(
    values: Tensor | Sequence[float],
    *,
    name: str,
    mode_count: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.shape != (mode_count,):
        raise ValueError(f"{name} must be a length-{mode_count} vector")
    if bool(torch.any(~torch.isfinite(tensor))) or bool(torch.any(tensor < 0)):
        raise ValueError(f"{name} must be finite and non-negative")
    total = tensor.sum()
    if float(total) <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return tensor / total


def _shift_vector(
    value: Tensor | Sequence[float] | None,
    mode_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    if value is None:
        return torch.zeros(mode_count, dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.shape != (mode_count,) or bool(torch.any(~torch.isfinite(tensor))):
        raise ValueError(f"component shifts must be a finite length-{mode_count} vector")
    return tensor


@dataclass(frozen=True)
class ModeMixtureProblem:
    """All objective probabilities plus geometry used only for rendering."""

    truth_weights: Tensor
    teacher_weights: Tensor
    initial_student_weights: Tensor
    capacity_mask: Tensor
    omitted_teacher_modes: tuple[int, ...]
    base_mode_means: Tensor
    truth_mode_means: Tensor
    teacher_mode_means: Tensor
    student_mode_means: Tensor
    component_std: float
    omission_floor: float

    def __post_init__(self) -> None:
        n = self.truth_weights.numel()
        vectors = (
            self.truth_weights,
            self.teacher_weights,
            self.initial_student_weights,
            self.base_mode_means,
            self.truth_mode_means,
            self.teacher_mode_means,
            self.student_mode_means,
        )
        if n < 2 or any(vector.shape != (n,) for vector in vectors):
            raise ValueError("all probability/mean vectors must share length >= 2")
        if self.capacity_mask.shape != (n,) or self.capacity_mask.dtype != torch.bool:
            raise ValueError("capacity_mask must be a boolean vector matching the modes")
        if not bool(torch.any(self.capacity_mask)):
            raise ValueError("capacity mask must retain at least one mode")
        if len({value.dtype for value in vectors}) != 1 or len({value.device for value in vectors}) != 1:
            raise ValueError("problem tensors must share dtype and device")
        for name, weights, allow_zero in (
            ("truth", self.truth_weights, False),
            ("teacher", self.teacher_weights, True),
            ("student", self.initial_student_weights, True),
        ):
            if bool(torch.any(~torch.isfinite(weights))) or bool(torch.any(weights < 0)):
                raise ValueError(f"{name} weights must be finite and non-negative")
            if not torch.allclose(weights.sum(), weights.new_tensor(1.0), atol=1e-12, rtol=1e-12):
                raise ValueError(f"{name} weights must sum to one")
            if not allow_zero and bool(torch.any(weights == 0)):
                raise ValueError("truth weights must be strictly positive")
        if bool(torch.any(self.initial_student_weights[self.capacity_mask] <= 0)):
            raise ValueError("every reachable student mode must have epsilon support")
        if bool(torch.any(self.initial_student_weights[~self.capacity_mask] != 0)):
            raise ValueError("structurally masked modes must have exactly zero student mass")
        if any(bool(torch.any(~torch.isfinite(means))) for means in vectors[3:]):
            raise ValueError("rendered means must be finite")
        expected_omissions = tuple(int(index) for index in torch.nonzero(self.teacher_weights == 0).flatten())
        if not set(expected_omissions).issubset(self.omitted_teacher_modes):
            raise ValueError("every exact teacher zero must be recorded as an omitted mode")
        if self.omission_floor == 0.0 and self.omitted_teacher_modes != expected_omissions:
            raise ValueError("exact omissions must match zero teacher weights")
        if self.component_std <= 0.0 or self.omission_floor < 0.0:
            raise ValueError("component_std must be positive and omission_floor non-negative")

    @property
    def mode_count(self) -> int:
        return self.truth_weights.numel()

    @property
    def capacity_modes(self) -> tuple[int, ...]:
        return tuple(int(index) for index in torch.nonzero(self.capacity_mask).flatten())

    @property
    def has_hard_capacity_mask(self) -> bool:
        return not bool(torch.all(self.capacity_mask))

    @property
    def exact_teacher_omission(self) -> bool:
        return bool(torch.any(self.teacher_weights == 0))

    def means(self, distribution: DistributionName) -> Tensor:
        if distribution == "truth":
            return self.truth_mode_means
        if distribution == "teacher":
            return self.teacher_mode_means
        if distribution == "student":
            return self.student_mode_means
        raise ValueError(f"unknown distribution {distribution!r}")


def _initial_student_distribution(
    *,
    mode_count: int,
    capacity_mask: Tensor,
    initial_student_weights: Tensor | Sequence[float] | None,
    initial_student_modes: Sequence[int] | None,
    support_epsilon: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    reachable_count = int(capacity_mask.sum())
    if support_epsilon <= 0.0 or reachable_count * support_epsilon >= 1.0:
        raise ValueError("support_epsilon must lie in (0, 1 / reachable_mode_count)")
    if initial_student_weights is not None and initial_student_modes is not None:
        raise ValueError("specify initial weights or initial modes, not both")
    if initial_student_weights is not None:
        raw = torch.as_tensor(initial_student_weights, dtype=dtype, device=device)
        if raw.shape != (mode_count,):
            raise ValueError(f"initial_student_weights must have length {mode_count}")
        if bool(torch.any(raw[~capacity_mask] != 0)):
            raise ValueError("initial weights cannot place mass outside the hard capacity mask")
        reachable = _normalized_nonnegative(
            raw[capacity_mask],
            name="reachable initial student weights",
            mode_count=reachable_count,
            dtype=dtype,
            device=device,
        ).clamp_min(support_epsilon)
        reachable = reachable / reachable.sum()
    elif initial_student_modes is None:
        reachable = torch.full(
            (reachable_count,),
            1.0 / reachable_count,
            dtype=dtype,
            device=device,
        )
    else:
        modes = tuple(int(index) for index in initial_student_modes)
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("initial_student_modes must be non-empty and unique")
        if any(index < 0 or index >= mode_count or not bool(capacity_mask[index]) for index in modes):
            raise ValueError("initial student modes must be reachable valid indices")
        reachable = torch.full((reachable_count,), support_epsilon, dtype=dtype, device=device)
        capacity_indices = torch.nonzero(capacity_mask).flatten()
        active_positions = [int(torch.nonzero(capacity_indices == mode).item()) for mode in modes]
        active_mass = (1.0 - support_epsilon * (reachable_count - len(modes))) / len(modes)
        reachable[active_positions] = active_mass
    result = torch.zeros(mode_count, dtype=dtype, device=device)
    result[capacity_mask] = reachable
    return result


def build_decaying_problem(
    *,
    mode_count: int = 5,
    decay: float = 0.65,
    teacher_omitted_modes: Sequence[int] | None = None,
    omission_floor: float = 1e-6,
    teacher_weight_power: float = 0.75,
    teacher_weight_multipliers: Tensor | Sequence[float] | None = None,
    teacher_weights: Tensor | Sequence[float] | None = None,
    initial_student_weights: Tensor | Sequence[float] | None = None,
    initial_student_modes: Sequence[int] | None = None,
    support_epsilon: float = 5e-4,
    student_capacity_modes: Sequence[int] | None = None,
    mode_spacing: float = 4.0,
    component_std: float = 0.55,
    global_shift: float = 0.0,
    component_shifts: Tensor | Sequence[float] | None = None,
    teacher_global_shift: float = 0.0,
    teacher_component_shifts: Tensor | Sequence[float] | None = None,
    student_global_shift: float = 0.0,
    student_component_shifts: Tensor | Sequence[float] | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | str = "cpu",
) -> ModeMixtureProblem:
    """Build a decaying truth and independently distorted teacher/student.

    ``teacher_omitted_modes=None`` omits the final mode.  A positive
    ``omission_floor`` makes this an effective omission with finite RKL;
    ``omission_floor=0`` creates exact zero support.  ``initial_student_modes``
    only controls epsilon initialization.  ``student_capacity_modes`` is the
    separate structural mask.
    """

    if not isinstance(mode_count, int) or isinstance(mode_count, bool) or mode_count < 2:
        raise ValueError("mode_count must be an integer >= 2")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie in (0, 1) so the largest truth mode is unique")
    if omission_floor < 0.0 or omission_floor * mode_count >= 1.0:
        raise ValueError("omission_floor must lie in [0, 1 / mode_count)")
    if teacher_weight_power <= 0.0:
        raise ValueError("teacher_weight_power must be positive")
    if mode_spacing <= 0.0 or component_std <= 0.0:
        raise ValueError("mode_spacing and component_std must be positive")
    global_values = (global_shift, teacher_global_shift, student_global_shift)
    if any(not torch.isfinite(torch.tensor(value)) for value in global_values):
        raise ValueError("global shifts must be finite")

    omissions = (mode_count - 1,) if teacher_omitted_modes is None else tuple(
        sorted(int(index) for index in teacher_omitted_modes)
    )
    if len(set(omissions)) != len(omissions) or any(index < 0 or index >= mode_count for index in omissions):
        raise ValueError("teacher omissions must be unique valid indices")
    if len(omissions) == mode_count:
        raise ValueError("teacher cannot omit every mode")

    indices = torch.arange(mode_count, dtype=dtype, device=device)
    truth = decay**indices
    truth = truth / truth.sum()
    if teacher_weights is None:
        raw_teacher = truth.pow(teacher_weight_power)
        if teacher_weight_multipliers is not None:
            multipliers = torch.as_tensor(teacher_weight_multipliers, dtype=dtype, device=device)
            if multipliers.shape != (mode_count,) or bool(torch.any(~torch.isfinite(multipliers) | (multipliers < 0))):
                raise ValueError("teacher multipliers must be a finite non-negative mode vector")
            raw_teacher = raw_teacher * multipliers
    else:
        raw_teacher = torch.as_tensor(teacher_weights, dtype=dtype, device=device)
        if raw_teacher.shape != (mode_count,):
            raise ValueError(f"teacher_weights must have length {mode_count}")
    raw_teacher = raw_teacher.clone()
    if omissions:
        raw_teacher[list(omissions)] = omission_floor
    actual_zero_modes = tuple(int(index) for index in torch.nonzero(raw_teacher == 0).flatten())
    effective_omissions = tuple(sorted(set(omissions).union(actual_zero_modes)))
    teacher = _normalized_nonnegative(
        raw_teacher,
        name="teacher weights",
        mode_count=mode_count,
        dtype=dtype,
        device=device,
    )

    capacity_mask = torch.zeros(mode_count, dtype=torch.bool, device=device)
    if student_capacity_modes is None:
        capacity_mask[:] = True
    else:
        capacity = tuple(int(index) for index in student_capacity_modes)
        if not capacity or len(set(capacity)) != len(capacity):
            raise ValueError("student_capacity_modes must be non-empty and unique")
        if any(index < 0 or index >= mode_count for index in capacity):
            raise ValueError("student capacity mode index is out of range")
        capacity_mask[list(capacity)] = True
    student = _initial_student_distribution(
        mode_count=mode_count,
        capacity_mask=capacity_mask,
        initial_student_weights=initial_student_weights,
        initial_student_modes=initial_student_modes,
        support_epsilon=support_epsilon,
        dtype=dtype,
        device=device,
    )

    base_means = mode_spacing * (indices - 0.5 * (mode_count - 1)) + global_shift
    common_component_shifts = _shift_vector(component_shifts, mode_count, dtype=dtype, device=device)
    teacher_shifts = _shift_vector(teacher_component_shifts, mode_count, dtype=dtype, device=device)
    student_shifts = _shift_vector(student_component_shifts, mode_count, dtype=dtype, device=device)
    truth_means = base_means + common_component_shifts
    teacher_means = truth_means + teacher_global_shift + teacher_shifts
    student_means = truth_means + student_global_shift + student_shifts
    return ModeMixtureProblem(
        truth_weights=truth,
        teacher_weights=teacher,
        initial_student_weights=student,
        capacity_mask=capacity_mask,
        omitted_teacher_modes=effective_omissions,
        base_mode_means=base_means,
        truth_mode_means=truth_means,
        teacher_mode_means=teacher_means,
        student_mode_means=student_means,
        component_std=float(component_std),
        omission_floor=float(omission_floor),
    )


class ModePolicy(nn.Module):
    """Categorical logits only for structurally reachable modes."""

    def __init__(self, problem: ModeMixtureProblem) -> None:
        super().__init__()
        self.register_buffer("capacity_mask", problem.capacity_mask.clone())
        self.logits = nn.Parameter(problem.initial_student_weights[self.capacity_mask].log().clone())

    @property
    def active_log_probs(self) -> Tensor:
        return self.logits.log_softmax(dim=-1)

    @property
    def full_log_probs(self) -> Tensor:
        result = torch.full(
            self.capacity_mask.shape,
            -torch.inf,
            dtype=self.logits.dtype,
            device=self.logits.device,
        )
        return result.scatter(0, torch.nonzero(self.capacity_mask).flatten(), self.active_log_probs)

    @property
    def full_probs(self) -> Tensor:
        return self.full_log_probs.exp()


class UndefinedPopulationObjective(ValueError):
    """Raised when an exact support mismatch makes an objective infinite."""


def _conditional_target(weights: Tensor, capacity_mask: Tensor) -> tuple[Tensor, float]:
    retained = weights[capacity_mask].sum()
    if float(retained) <= 0.0:
        raise UndefinedPopulationObjective("the student capacity excludes the entire target support")
    return weights[capacity_mask] / retained, float(retained)


def population_loss(
    method: Method,
    policy: ModePolicy,
    problem: ModeMixtureProblem,
    *,
    entropy_alpha: float = 1.0,
) -> Tensor:
    """Evaluate a direct shared categorical population objective."""

    if entropy_alpha < 0.0:
        raise ValueError("entropy_alpha must be non-negative")
    if method in {"sft", "opd_fkl"}:
        excluded_teacher_mass = problem.teacher_weights[~policy.capacity_mask].sum()
        if float(excluded_teacher_mass) > 0.0:
            raise UndefinedPopulationObjective(
                "forward teacher objective is infinite because hard student capacity excludes teacher mass"
            )
        teacher, _ = _conditional_target(problem.teacher_weights, policy.capacity_mask)
        if method == "sft":
            return sft_population_loss(policy.logits, teacher)
        return opd_fkl_loss(FKLBatch(policy.active_log_probs, teacher)).loss
    if method == "opd_rkl":
        teacher, retained_mass = _conditional_target(problem.teacher_weights, policy.capacity_mask)
        if bool(torch.any(teacher == 0)):
            raise UndefinedPopulationObjective(
                "reverse KL is infinite: reachable student modes have exact zero teacher support; "
                "use a positive omission_floor or a matching hard capacity mask"
            )
        # Conditioning lets the shared operator consume a normalized target;
        # restoring -log q(capacity) makes the reported value the full KL.
        return opd_rkl_population_loss(policy.logits, teacher) - policy.logits.new_tensor(
            retained_mass
        ).log()
    truth_log_reward = problem.truth_weights[policy.capacity_mask].log()
    if method == "truth_rl":
        return rl_population_loss(policy.logits, truth_log_reward)
    if method == "truth_rl_entropy":
        return rl_population_loss(policy.logits, truth_log_reward, entropy_bonus=entropy_alpha)
    raise ValueError(f"unknown method {method!r}")


@dataclass(frozen=True)
class EndpointPredictions:
    truth_weights: tuple[float, ...]
    teacher_weights: tuple[float, ...]
    initial_student_weights: tuple[float, ...]
    capacity_modes: tuple[int, ...]
    omitted_teacher_modes: tuple[int, ...]
    omitted_truth_weight: float
    pure_truth_rl_selected_mode: int
    sft_fkl_endpoint: tuple[float, ...] | None
    opd_rkl_endpoint: tuple[float, ...] | None
    entropy_truth_rl_endpoint: tuple[float, ...]
    pure_truth_rl_endpoint: tuple[float, ...]
    sft_fkl_available: bool
    opd_rkl_available: bool
    support_note: str


def _embed_active(active: Tensor, capacity_mask: Tensor) -> tuple[float, ...]:
    full = torch.zeros(capacity_mask.shape, dtype=active.dtype, device=active.device)
    full[capacity_mask] = active
    return tuple(float(value) for value in full)


def theory_predictions(
    problem: ModeMixtureProblem,
    *,
    entropy_alpha: float = 1.0,
) -> EndpointPredictions:
    """Return categorical modal-weight endpoints before optimization.

    These predictions do not include the optional shared-translation
    bottleneck, whose joint labeled objective can reweight modes.
    """

    if entropy_alpha < 0.0:
        raise ValueError("entropy_alpha must be non-negative")
    teacher_active, teacher_retained = _conditional_target(problem.teacher_weights, problem.capacity_mask)
    truth_active, _ = _conditional_target(problem.truth_weights, problem.capacity_mask)
    forward_available = abs(teacher_retained - 1.0) <= 1e-12
    rkl_available = bool(torch.all(teacher_active > 0))
    selected_active = int(truth_active.argmax())
    capacity_indices = torch.nonzero(problem.capacity_mask).flatten()
    selected_mode = int(capacity_indices[selected_active])
    pure_active = torch.zeros_like(truth_active)
    pure_active[selected_active] = 1.0
    teacher_endpoint = _embed_active(teacher_active, problem.capacity_mask)
    if entropy_alpha == 0.0:
        entropy_active = pure_active
    else:
        # The tempered optimum is proportional to
        # ``truth_active ** (1 / entropy_alpha)``.  Compute it in log space so
        # small positive temperatures do not turn every raw power into zero.
        entropy_active = (truth_active.log() / entropy_alpha).softmax(dim=-1)
    entropy_endpoint = _embed_active(entropy_active, problem.capacity_mask)
    omitted_truth_weight = (
        float(problem.truth_weights[list(problem.omitted_teacher_modes)].sum())
        if problem.omitted_teacher_modes
        else 0.0
    )
    return EndpointPredictions(
        truth_weights=tuple(float(value) for value in problem.truth_weights),
        teacher_weights=tuple(float(value) for value in problem.teacher_weights),
        initial_student_weights=tuple(float(value) for value in problem.initial_student_weights),
        capacity_modes=problem.capacity_modes,
        omitted_teacher_modes=problem.omitted_teacher_modes,
        omitted_truth_weight=omitted_truth_weight,
        pure_truth_rl_selected_mode=selected_mode,
        sft_fkl_endpoint=teacher_endpoint if forward_available else None,
        opd_rkl_endpoint=teacher_endpoint if rkl_available else None,
        entropy_truth_rl_endpoint=entropy_endpoint,
        pure_truth_rl_endpoint=_embed_active(pure_active, problem.capacity_mask),
        sft_fkl_available=forward_available,
        opd_rkl_available=rkl_available,
        support_note=(
            "Epsilon-initialized reachable modes can be recovered; modes outside capacity are structural zeros. "
            "Exact teacher zeros make reverse KL infinite unless those modes are also structurally excluded."
        ),
    )


@dataclass(frozen=True)
class ModeMetrics:
    """Categorical metrics.

    ``truth_support_recall`` uses the shared numerical mass threshold
    ``1e-6``; it is effective coverage, not literal softmax support.
    """

    student_weights: tuple[float, ...]
    entropy: float
    effective_support: float
    expected_truth_log_weight: float
    kl_truth_student: float
    kl_student_truth: float
    kl_teacher_student: float
    kl_student_teacher: float
    truth_support_recall: float
    student_omitted_teacher_mass: float
    largest_student_mode: int


@torch.no_grad()
def evaluate_policy(policy: ModePolicy, problem: ModeMixtureProblem) -> ModeMetrics:
    student_log_probs = policy.full_log_probs
    teacher_log_probs = torch.where(
        problem.teacher_weights > 0,
        problem.teacher_weights.log(),
        torch.full_like(problem.teacher_weights, -torch.inf),
    )
    truth_log_probs = problem.truth_weights.log()
    shared = categorical_metrics(
        student_log_probs,
        teacher_log_probs=teacher_log_probs,
        truth_log_probs=truth_log_probs,
        utility=truth_log_probs,
    )
    student = student_log_probs.exp()
    omitted_mass = (
        student[list(problem.omitted_teacher_modes)].sum()
        if problem.omitted_teacher_modes
        else student.new_zeros(())
    )
    return ModeMetrics(
        student_weights=tuple(float(value) for value in student),
        entropy=float(shared["entropy"]),
        effective_support=float(shared["effective_support"]),
        expected_truth_log_weight=float(shared["expected_utility"]),
        kl_truth_student=float(shared["kl_truth_student"]),
        kl_student_truth=float(shared["kl_student_truth"]),
        kl_teacher_student=float(shared["kl_teacher_student"]),
        kl_student_teacher=float(shared["kl_student_teacher"]),
        truth_support_recall=float(shared["truth_support_recall"]),
        student_omitted_teacher_mass=float(omitted_mass),
        largest_student_mode=int(student.argmax()),
    )


@dataclass(frozen=True)
class TrajectoryConfig:
    steps: int = 40
    learning_rate: float = 0.20
    record_every: int = 10
    entropy_alpha: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(self.record_every, int) or self.record_every < 1:
            raise ValueError("record_every must be a positive integer")
        if self.learning_rate <= 0.0 or self.entropy_alpha < 0.0:
            raise ValueError("learning rate must be positive and entropy alpha non-negative")


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    loss: float
    metrics: ModeMetrics

    def as_record(self, method: Method) -> dict[str, object]:
        record: dict[str, object] = {"method": method, "step": self.step, "loss": self.loss}
        record.update(asdict(self.metrics))
        return record


@dataclass(frozen=True)
class PopulationTrajectory:
    method: Method
    config: TrajectoryConfig
    points: tuple[TrajectoryPoint, ...]

    @property
    def final(self) -> TrajectoryPoint:
        return self.points[-1]

    def records(self) -> list[dict[str, object]]:
        return [point.as_record(self.method) for point in self.points]


@dataclass(frozen=True)
class PopulationStudy:
    predictions: EndpointPredictions
    trajectories: tuple[PopulationTrajectory, ...]

    def trajectory(self, method: Method) -> PopulationTrajectory:
        matches = [trajectory for trajectory in self.trajectories if trajectory.method == method]
        if len(matches) != 1:
            raise KeyError(f"study does not contain exactly one {method!r} trajectory")
        return matches[0]

    def records(self) -> list[dict[str, object]]:
        return [record for trajectory in self.trajectories for record in trajectory.records()]


def run_population_trajectory(
    method: Method,
    *,
    problem: ModeMixtureProblem | None = None,
    config: TrajectoryConfig = TrajectoryConfig(),
) -> PopulationTrajectory:
    """Run a deterministic exact-population optimizer trajectory."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    problem = build_decaying_problem() if problem is None else problem
    policy = ModePolicy(problem)
    # Validate support before constructing the optimizer or recording a
    # misleading finite-looking initial point.
    initial_loss = population_loss(method, policy, problem, entropy_alpha=config.entropy_alpha)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    points: list[TrajectoryPoint] = []

    def record(step: int, loss_value: Tensor) -> None:
        points.append(
            TrajectoryPoint(
                step=step,
                loss=float(loss_value.detach()),
                metrics=evaluate_policy(policy, problem),
            )
        )

    record(0, initial_loss)
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_value = population_loss(method, policy, problem, entropy_alpha=config.entropy_alpha)
        loss_value.backward()
        optimizer.step()
        if step % config.record_every == 0 or step == config.steps:
            updated = population_loss(method, policy, problem, entropy_alpha=config.entropy_alpha)
            record(step, updated)
    return PopulationTrajectory(method=method, config=config, points=tuple(points))


def run_population_study(
    *,
    problem: ModeMixtureProblem | None = None,
    methods: Sequence[Method] = METHODS,
    config: TrajectoryConfig = TrajectoryConfig(),
) -> PopulationStudy:
    """Run all requested methods from the identical initial distribution."""

    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    problem = build_decaying_problem() if problem is None else problem
    predictions = theory_predictions(problem, entropy_alpha=config.entropy_alpha)
    trajectories: list[PopulationTrajectory] = []
    forward_points: tuple[TrajectoryPoint, ...] | None = None
    for method in methods:
        if method in {"sft", "opd_fkl"} and forward_points is not None:
            trajectory = PopulationTrajectory(method=method, config=config, points=forward_points)
        else:
            trajectory = run_population_trajectory(method, problem=problem, config=config)
            if method in {"sft", "opd_fkl"}:
                # These are the same population objective, not merely similar
                # endpoints. Reusing the path is both an invariant and avoids
                # paying twice for an identical deterministic computation.
                forward_points = trajectory.points
        trajectories.append(trajectory)
    return PopulationStudy(predictions=predictions, trajectories=tuple(trajectories))


class LabeledLocationPolicy(nn.Module):
    """Mode logits plus one learnable translation of all student kernels."""

    def __init__(self, problem: ModeMixtureProblem) -> None:
        super().__init__()
        self.mode_policy = ModePolicy(problem)
        self.translation = nn.Parameter(problem.truth_weights.new_zeros(()))

    @property
    def full_probs(self) -> Tensor:
        return self.mode_policy.full_probs

    def means(self, problem: ModeMixtureProblem) -> Tensor:
        return problem.student_mode_means + self.translation


def labeled_location_population_loss(
    method: Method,
    policy: LabeledLocationPolicy,
    problem: ModeMixtureProblem,
    *,
    entropy_alpha: float = 1.0,
) -> Tensor:
    """Categorical objective plus the exact labeled Gaussian location term.

    All components use the same fixed variance.  For matching labels,
    ``KL(N(mu_p,sigma) || N(mu_t,sigma))`` contributes
    ``(mu_p-mu_t)^2 / (2 sigma^2)``.  Forward objectives average this term
    under teacher mass; reverse/trusted-reward objectives average it under
    current student mass.
    """

    categorical = population_loss(
        method,
        policy.mode_policy,
        problem,
        entropy_alpha=entropy_alpha,
    )
    student_means = policy.means(problem)
    if method in {"sft", "opd_fkl"}:
        weights = problem.teacher_weights.detach()
        target_means = problem.teacher_mode_means
    elif method == "opd_rkl":
        weights = policy.full_probs
        target_means = problem.teacher_mode_means
    else:
        weights = policy.full_probs
        target_means = problem.truth_mode_means
    squared_standardized_shift = ((student_means - target_means) / problem.component_std).square()
    return categorical + 0.5 * (weights * squared_standardized_shift).sum()


@dataclass(frozen=True)
class LocationGradient:
    method: Method
    weighting_distribution: Literal["teacher", "student"]
    initial_translation: float
    loss: float
    gradient: float
    descent_direction: float
    component_gradients: tuple[float, ...]
    component_descent_directions: tuple[float, ...]


def exact_initial_location_gradient(
    method: Method,
    problem: ModeMixtureProblem,
    *,
    entropy_alpha: float = 1.0,
) -> LocationGradient:
    """Return the exact initial scalar shift gradient via the shared objective."""

    policy = LabeledLocationPolicy(problem)
    loss_value = labeled_location_population_loss(
        method,
        policy,
        problem,
        entropy_alpha=entropy_alpha,
    )
    gradient = torch.autograd.grad(loss_value, policy.translation)[0]
    weighting: Literal["teacher", "student"] = (
        "teacher" if method in {"sft", "opd_fkl"} else "student"
    )
    if method in {"sft", "opd_fkl"}:
        weights = problem.teacher_weights
        targets = problem.teacher_mode_means
    elif method == "opd_rkl":
        weights = policy.full_probs.detach()
        targets = problem.teacher_mode_means
    else:
        weights = policy.full_probs.detach()
        targets = problem.truth_mode_means
    component_gradients = (
        weights * (problem.student_mode_means - targets) / (problem.component_std**2)
    )
    return LocationGradient(
        method=method,
        weighting_distribution=weighting,
        initial_translation=float(policy.translation.detach()),
        loss=float(loss_value.detach()),
        gradient=float(gradient.detach()),
        descent_direction=float(-gradient.detach()),
        component_gradients=tuple(float(value) for value in component_gradients),
        component_descent_directions=tuple(float(-value) for value in component_gradients),
    )


@dataclass(frozen=True)
class LabeledLocationMetrics:
    translation: float
    student_means: tuple[float, ...]
    student_weighted_truth_location_rmse: float
    teacher_weighted_teacher_location_rmse: float
    mode_metrics: ModeMetrics


@torch.no_grad()
def evaluate_labeled_location_policy(
    policy: LabeledLocationPolicy,
    problem: ModeMixtureProblem,
) -> LabeledLocationMetrics:
    student_means = policy.means(problem)
    student_truth_rmse = torch.sqrt(
        (policy.full_probs * (student_means - problem.truth_mode_means).square()).sum()
    )
    teacher_rmse = torch.sqrt(
        (problem.teacher_weights * (student_means - problem.teacher_mode_means).square()).sum()
    )
    return LabeledLocationMetrics(
        translation=float(policy.translation),
        student_means=tuple(float(value) for value in student_means),
        student_weighted_truth_location_rmse=float(student_truth_rmse),
        teacher_weighted_teacher_location_rmse=float(teacher_rmse),
        mode_metrics=evaluate_policy(policy.mode_policy, problem),
    )


@dataclass(frozen=True)
class LabeledLocationPoint:
    step: int
    loss: float
    metrics: LabeledLocationMetrics

    def as_record(self, method: Method) -> dict[str, object]:
        record: dict[str, object] = {
            "method": method,
            "step": self.step,
            "loss": self.loss,
            "translation": self.metrics.translation,
            "student_means": self.metrics.student_means,
            "student_weighted_truth_location_rmse": self.metrics.student_weighted_truth_location_rmse,
            "teacher_weighted_teacher_location_rmse": self.metrics.teacher_weighted_teacher_location_rmse,
        }
        record.update(asdict(self.metrics.mode_metrics))
        return record


@dataclass(frozen=True)
class LabeledLocationTrajectory:
    method: Method
    config: TrajectoryConfig
    points: tuple[LabeledLocationPoint, ...]

    @property
    def final(self) -> LabeledLocationPoint:
        return self.points[-1]

    def records(self) -> list[dict[str, object]]:
        return [point.as_record(self.method) for point in self.points]


def run_labeled_location_trajectory(
    method: Method,
    *,
    problem: ModeMixtureProblem | None = None,
    config: TrajectoryConfig = TrajectoryConfig(),
) -> LabeledLocationTrajectory:
    """Run the optional scalar-translation trajectory with exact gradients."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    problem = build_decaying_problem() if problem is None else problem
    policy = LabeledLocationPolicy(problem)
    initial_loss = labeled_location_population_loss(
        method,
        policy,
        problem,
        entropy_alpha=config.entropy_alpha,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    points: list[LabeledLocationPoint] = []

    def record(step: int, loss_value: Tensor) -> None:
        points.append(
            LabeledLocationPoint(
                step=step,
                loss=float(loss_value.detach()),
                metrics=evaluate_labeled_location_policy(policy, problem),
            )
        )

    record(0, initial_loss)
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_value = labeled_location_population_loss(
            method,
            policy,
            problem,
            entropy_alpha=config.entropy_alpha,
        )
        loss_value.backward()
        optimizer.step()
        if step % config.record_every == 0 or step == config.steps:
            updated = labeled_location_population_loss(
                method,
                policy,
                problem,
                entropy_alpha=config.entropy_alpha,
            )
            record(step, updated)
    return LabeledLocationTrajectory(method=method, config=config, points=tuple(points))


@dataclass(frozen=True)
class RenderedMixtures:
    points: Tensor
    truth_density: Tensor
    teacher_density: Tensor
    student_density: Tensor


def _render_density(weights: Tensor, means: Tensor, std: float, points: Tensor) -> Tensor:
    log_components = normal_log_prob(
        points.unsqueeze(-1),
        means,
        torch.full_like(means, torch.log(means.new_tensor(std))),
    )
    log_weights = torch.where(weights > 0, weights.log(), torch.full_like(weights, -torch.inf))
    return torch.logsumexp(log_weights + log_components, dim=-1).exp()


@torch.no_grad()
def render_mixtures(
    problem: ModeMixtureProblem,
    *,
    student_weights: Tensor | Sequence[float] | None = None,
    points: Tensor | None = None,
    points_per_std: int = 12,
    padding_stds: float = 5.0,
) -> RenderedMixtures:
    """Render fixed Gaussian kernels without changing categorical training."""

    if points_per_std < 2 or padding_stds <= 0.0:
        raise ValueError("points_per_std >= 2 and positive padding are required")
    dtype, device = problem.truth_weights.dtype, problem.truth_weights.device
    if student_weights is None:
        student = problem.initial_student_weights
    else:
        student = _normalized_nonnegative(
            student_weights,
            name="student render weights",
            mode_count=problem.mode_count,
            dtype=dtype,
            device=device,
        )
    if points is None:
        all_means = torch.cat(
            (problem.truth_mode_means, problem.teacher_mode_means, problem.student_mode_means)
        )
        lower = float(all_means.min() - padding_stds * problem.component_std)
        upper = float(all_means.max() + padding_stds * problem.component_std)
        count = max(3, int((upper - lower) * points_per_std / problem.component_std) + 1)
        points = torch.linspace(lower, upper, count, dtype=dtype, device=device)
    elif points.ndim != 1 or points.numel() < 2:
        raise ValueError("render points must be a one-dimensional vector of length >= 2")
    return RenderedMixtures(
        points=points,
        truth_density=_render_density(
            problem.truth_weights,
            problem.truth_mode_means,
            problem.component_std,
            points,
        ),
        teacher_density=_render_density(
            problem.teacher_weights,
            problem.teacher_mode_means,
            problem.component_std,
            points,
        ),
        student_density=_render_density(
            student,
            problem.student_mode_means,
            problem.component_std,
            points,
        ),
    )
