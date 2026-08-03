"""First-principles asymmetric-mixture comparison of SFT, OPD, and RL.

The deliberately capacity-limited student is one Gaussian.  Its teacher is

``q = 0.8 Normal(-4, 1) + 0.2 Normal(+4, 0.1)``.

This one choice separates three notions that are otherwise easy to conflate:

* SFT / forward KL projects onto the teacher's first two moments, so it covers
  both modes with ``mean=-2.4`` and ``variance=11.042``.
* reverse-KL OPD prefers the broad, high-mass left component;
* pure RL with utility ``log q(x)`` prefers the narrow right density peak and
  drives the student's variance toward zero.

For entropy-regularized RL, ``J_alpha=E_p[log q] + alpha H(p)``, isolated-mode
asymptotics predict a global switch at ``alpha=0.397940...``.  The full
deterministic landscape is searched before optimizer paths are run, so a
local basin reached from one initialization is never mistaken for the global
objective geometry.

All update losses and population values come from ``toy_methods``.  This file
only specifies distributions, collection, sweeps, and reported metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal, Sequence

import torch
from torch import Tensor

try:  # Support both package imports and adding this examples directory to sys.path.
    from .families import (
        DEFAULT_DTYPE,
        GaussianMixture1D,
        GaussianPolicy1D,
        QuadratureGrid1D,
        normal_log_prob,
    )
    from .toy_methods import (
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_metrics,
        opd_rkl_loss,
        opd_rkl_population_values,
        rl_loss,
        rl_population_values,
        sft_loss,
        sft_population_values,
    )
except ImportError:  # pragma: no cover - used by standalone example scripts
    from families import DEFAULT_DTYPE, GaussianMixture1D, GaussianPolicy1D, QuadratureGrid1D, normal_log_prob
    from toy_methods import (
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_metrics,
        opd_rkl_loss,
        opd_rkl_population_values,
        rl_loss,
        rl_population_values,
        sft_loss,
        sft_population_values,
    )


TrajectoryMethod = Literal["sft", "opd_rkl", "rl"]
Basin = Literal["all", "left", "right"]


def asymmetric_teacher(
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | str = "cpu",
) -> GaussianMixture1D:
    """The smallest mixture where probability mass and density peak disagree."""

    return GaussianMixture1D.from_parameters(
        weights=(0.8, 0.2),
        means=(-4.0, 4.0),
        stds=(1.0, 0.1),
        dtype=dtype,
        device=device,
    )


@dataclass(frozen=True)
class TheoryPredictions:
    sft_mean: float
    sft_variance: float
    sft_std: float
    left_peak_log_density: float
    right_peak_log_density: float
    pure_rl_mode: str
    opd_global_mode: str
    entropy_switch_alpha: float


def isolated_mode_switch_alpha(
    teacher: GaussianMixture1D,
    *,
    left_component: int = 0,
    right_component: int = 1,
) -> float:
    r"""Predict the entropy-RL mode switch when components do not overlap.

    Near component ``k``, optimizing a Gaussian policy gives
    ``std_p=sqrt(alpha)*std_k``.  Constants cancel between modes, leaving

    ``log(w_r/w_l) + (alpha-1) log(std_r/std_l) = 0``.
    """

    weight_ratio = float(teacher.weights[right_component] / teacher.weights[left_component])
    std_ratio = float(teacher.stds[right_component] / teacher.stds[left_component])
    if weight_ratio <= 0.0 or std_ratio <= 0.0 or std_ratio == 1.0:
        raise ValueError("mode-switch prediction requires positive weights and unequal standard deviations")
    return 1.0 - log(weight_ratio) / log(std_ratio)


def theory_predictions(teacher: GaussianMixture1D | None = None) -> TheoryPredictions:
    """Return predictions fixed before looking at any optimizer trajectory."""

    teacher = asymmetric_teacher() if teacher is None else teacher
    component_peaks = teacher.log_prob(teacher.means)
    pure_rl_index = int(component_peaks.argmax())
    opd_index = int(teacher.weights.argmax())
    return TheoryPredictions(
        sft_mean=float(teacher.mean),
        sft_variance=float(teacher.variance),
        sft_std=sqrt(float(teacher.variance)),
        left_peak_log_density=float(component_peaks[0]),
        right_peak_log_density=float(component_peaks[1]),
        pure_rl_mode="left" if pure_rl_index == 0 else "right",
        opd_global_mode="left" if opd_index == 0 else "right",
        entropy_switch_alpha=isolated_mode_switch_alpha(teacher),
    )


@dataclass(frozen=True)
class LandscapeConfig:
    mean_min: float = -5.5
    mean_max: float = 5.0
    mean_count: int = 211
    min_std: float = 0.02
    max_std: float = 4.5
    log_std_count: int = 181
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if not self.mean_min < self.mean_max:
            raise ValueError("mean_min must be smaller than mean_max")
        if self.mean_count < 2 or self.log_std_count < 2:
            raise ValueError("landscape axes must contain at least two points")
        if not 0.0 < self.min_std < self.max_std:
            raise ValueError("require 0 < min_std < max_std")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class LandscapeMinimum:
    objective: str
    alpha: float | None
    basin: Basin
    mean: float
    std: float
    value: float
    mean_index: int
    log_std_index: int
    on_search_boundary: bool


@dataclass(frozen=True)
class ObjectiveLandscape:
    """Population objectives on a common ``(mean, log_std)`` search grid."""

    means: Tensor
    log_stds: Tensor
    sft_fkl: Tensor
    opd_rkl: Tensor
    pure_rl: Tensor
    entropy_rl_alpha_one: Tensor

    def __post_init__(self) -> None:
        expected = (self.means.numel(), self.log_stds.numel())
        for name in ("sft_fkl", "opd_rkl", "pure_rl", "entropy_rl_alpha_one"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")

    def entropy_rl(self, alpha: float) -> Tensor:
        """``-E[log q] - alpha H`` by linearity from two shared RL calls."""

        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        return self.pure_rl + alpha * (self.entropy_rl_alpha_one - self.pure_rl)

    def values(self, objective: str, *, alpha: float | None = None) -> Tensor:
        if objective in {"sft", "fkl", "sft_fkl"}:
            return self.sft_fkl
        if objective in {"opd", "opd_rkl", "rkl"}:
            return self.opd_rkl
        if objective in {"rl", "pure_rl"}:
            if alpha not in (None, 0.0):
                raise ValueError("pure RL has alpha=0; use objective='entropy_rl'")
            return self.pure_rl
        if objective in {"entropy_rl", "regularized_rl"}:
            if alpha is None:
                raise ValueError("entropy_rl requires alpha")
            return self.entropy_rl(alpha)
        raise ValueError(f"unknown objective {objective!r}")

    def minimum(
        self,
        objective: str,
        *,
        alpha: float | None = None,
        basin: Basin = "all",
    ) -> LandscapeMinimum:
        values = self.values(objective, alpha=alpha)
        if basin == "all":
            allowed_rows = torch.ones_like(self.means, dtype=torch.bool)
        elif basin == "left":
            allowed_rows = self.means < 0.0
        elif basin == "right":
            allowed_rows = self.means >= 0.0
        else:
            raise ValueError(f"unknown basin {basin!r}")
        if not bool(torch.any(allowed_rows)):
            raise ValueError(f"landscape has no points in the {basin!r} basin")
        masked = torch.where(allowed_rows[:, None], values, torch.inf)
        flat_index = int(masked.argmin())
        mean_index = flat_index // self.log_stds.numel()
        std_index = flat_index % self.log_stds.numel()
        boundary = mean_index in {0, self.means.numel() - 1} or std_index in {0, self.log_stds.numel() - 1}
        return LandscapeMinimum(
            objective=objective,
            alpha=alpha,
            basin=basin,
            mean=float(self.means[mean_index]),
            std=exp(float(self.log_stds[std_index])),
            value=float(values[mean_index, std_index]),
            mean_index=mean_index,
            log_std_index=std_index,
            on_search_boundary=boundary,
        )


def _candidate_axes(config: LandscapeConfig, teacher: GaussianMixture1D) -> tuple[Tensor, Tensor]:
    means = torch.linspace(
        config.mean_min,
        config.mean_max,
        config.mean_count,
        dtype=teacher.means.dtype,
        device=teacher.means.device,
    )
    log_stds = torch.linspace(
        log(config.min_std),
        log(config.max_std),
        config.log_std_count,
        dtype=teacher.means.dtype,
        device=teacher.means.device,
    )
    return means, log_stds


@torch.no_grad()
def build_objective_landscape(
    teacher: GaussianMixture1D | None = None,
    *,
    grid: QuadratureGrid1D | None = None,
    config: LandscapeConfig = LandscapeConfig(),
) -> ObjectiveLandscape:
    """Enumerate every population objective before running local optimizers.

    ``grid.weights`` may be nonuniform.  Discrete KL is still a consistent
    density KL because the common cell width cancels in ``log(p_i/q_i)``.
    For differential entropy, the alpha-one RL call adds ``log(weight_i)`` to
    utility; this exactly removes the cell-width term introduced by discrete
    entropy.  Intermediate alpha values then follow by objective linearity.
    """

    teacher = asymmetric_teacher() if teacher is None else teacher
    grid = QuadratureGrid1D.composite(dtype=teacher.means.dtype, device=teacher.means.device) if grid is None else grid
    if grid.points.dtype != teacher.means.dtype or grid.points.device != teacher.means.device:
        raise ValueError("teacher and grid must share dtype and device")

    means, log_stds = _candidate_axes(config, teacher)
    mean_mesh, log_std_mesh = torch.meshgrid(means, log_stds, indexing="ij")
    flat_means = mean_mesh.reshape(-1)
    flat_log_stds = log_std_mesh.reshape(-1)
    result = {
        "sft_fkl": torch.empty_like(flat_means),
        "opd_rkl": torch.empty_like(flat_means),
        "pure_rl": torch.empty_like(flat_means),
        "entropy_rl_alpha_one": torch.empty_like(flat_means),
    }

    teacher_log_density = teacher.log_prob(grid.points)
    teacher_log_mass = grid.normalize_log_density(teacher_log_density)
    teacher_prob = teacher_log_mass.exp()
    log_weights = grid.weights.log()

    for start in range(0, flat_means.numel(), config.chunk_size):
        stop = min(start + config.chunk_size, flat_means.numel())
        candidate_log_density = normal_log_prob(
            grid.points[None, :],
            flat_means[start:stop, None],
            flat_log_stds[start:stop, None],
        )
        candidate_log_mass = grid.normalize_log_density(candidate_log_density)
        teacher_rows = teacher_prob.expand_as(candidate_log_mass)
        utility_rows = teacher_log_density.expand_as(candidate_log_mass)
        differential_entropy_utility = (teacher_log_density + log_weights).expand_as(candidate_log_mass)

        result["sft_fkl"][start:stop] = sft_population_values(candidate_log_mass, teacher_rows)
        result["opd_rkl"][start:stop] = opd_rkl_population_values(candidate_log_mass, teacher_rows)
        result["pure_rl"][start:stop] = rl_population_values(candidate_log_mass, utility_rows)
        result["entropy_rl_alpha_one"][start:stop] = rl_population_values(
            candidate_log_mass,
            differential_entropy_utility,
            entropy_bonus=1.0,
        )

    shape = (means.numel(), log_stds.numel())
    return ObjectiveLandscape(
        means=means,
        log_stds=log_stds,
        sft_fkl=result["sft_fkl"].reshape(shape),
        opd_rkl=result["opd_rkl"].reshape(shape),
        pure_rl=result["pure_rl"].reshape(shape),
        entropy_rl_alpha_one=result["entropy_rl_alpha_one"].reshape(shape),
    )


@dataclass(frozen=True)
class PhaseRow:
    alpha: float
    left_mean: float
    left_std: float
    left_value: float
    right_mean: float
    right_std: float
    right_value: float
    global_basin: str
    value_gap_right_minus_left: float


def entropy_phase_sweep(landscape: ObjectiveLandscape, alphas: Sequence[float]) -> list[PhaseRow]:
    """Compare independently minimized left and right basins for each alpha."""

    rows = []
    for alpha_value in alphas:
        alpha = float(alpha_value)
        left = landscape.minimum("entropy_rl", alpha=alpha, basin="left")
        right = landscape.minimum("entropy_rl", alpha=alpha, basin="right")
        gap = right.value - left.value
        rows.append(
            PhaseRow(
                alpha=alpha,
                left_mean=left.mean,
                left_std=left.std,
                left_value=left.value,
                right_mean=right.mean,
                right_std=right.std,
                right_value=right.value,
                global_basin="right" if gap < 0.0 else "left",
                value_gap_right_minus_left=gap,
            )
        )
    return rows


def numerical_entropy_switch(
    landscape: ObjectiveLandscape,
    *,
    lower: float = 0.2,
    upper: float = 0.7,
    tolerance: float = 1e-6,
    max_iterations: int = 60,
) -> float:
    """Bisect the crossing of the best left and right grid basins."""

    if not 0.0 <= lower < upper or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid switch-search settings")

    def gap(alpha: float) -> float:
        left = landscape.minimum("entropy_rl", alpha=alpha, basin="left")
        right = landscape.minimum("entropy_rl", alpha=alpha, basin="right")
        return right.value - left.value

    low_gap, high_gap = gap(lower), gap(upper)
    if low_gap == 0.0:
        return lower
    if high_gap == 0.0:
        return upper
    if low_gap * high_gap > 0.0:
        raise ValueError("switch is not bracketed")
    for _ in range(max_iterations):
        middle = 0.5 * (lower + upper)
        middle_gap = gap(middle)
        if abs(middle_gap) <= tolerance or upper - lower <= tolerance:
            return middle
        if low_gap * middle_gap <= 0.0:
            upper, high_gap = middle, middle_gap
        else:
            lower, low_gap = middle, middle_gap
    return 0.5 * (lower + upper)


@dataclass(frozen=True)
class PolicyMetrics:
    mean: float
    std: float
    variance: float
    differential_entropy: float
    expected_log_teacher: float
    kl_teacher_student: float
    kl_student_teacher: float
    left_mass: float
    right_mass: float


@torch.no_grad()
def evaluate_policy(
    policy: GaussianPolicy1D,
    teacher: GaussianMixture1D,
    grid: QuadratureGrid1D,
    *,
    split: float = 0.0,
) -> PolicyMetrics:
    """Report direction-labelled KLs, utility, entropy, and mode mass."""

    student_log_mass = policy.grid_log_mass(grid)
    teacher_log_mass = teacher.grid_log_mass(grid)
    shared = categorical_metrics(
        student_log_mass,
        teacher_log_probs=teacher_log_mass,
        utility=teacher.log_prob(grid.points),
    )
    left_mass = policy.probability_left_of(split)
    return PolicyMetrics(
        mean=float(policy.mean),
        std=float(policy.std),
        variance=float(policy.variance),
        differential_entropy=float(policy.entropy()),
        expected_log_teacher=float(shared["expected_utility"]),
        kl_teacher_student=float(shared["kl_teacher_student"]),
        kl_student_teacher=float(shared["kl_student_teacher"]),
        left_mass=float(left_mass),
        right_mass=float(1.0 - left_mass),
    )


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    shared_surrogate_loss: float
    metrics: PolicyMetrics


@dataclass(frozen=True)
class OptimizerTrajectory:
    method: TrajectoryMethod
    alpha: float
    initial_mean: float
    initial_std: float
    points: tuple[TrajectoryPoint, ...]

    @property
    def final(self) -> TrajectoryPoint:
        return self.points[-1]


def _shared_sampled_loss(
    policy: GaussianPolicy1D,
    teacher: GaussianMixture1D,
    method: TrajectoryMethod,
    *,
    sample_count: int,
    alpha: float,
    demonstrations: Tensor | None,
) -> Tensor:
    if method == "sft":
        if demonstrations is None:
            raise ValueError("SFT requires fixed demonstrations")
        return sft_loss(SFTBatch(policy.log_prob(demonstrations))).loss

    sample = policy.deterministic_sample(sample_count)
    current_logp = policy.log_prob(sample.action)
    teacher_logp = teacher.log_prob(sample.action)
    if method == "opd_rkl":
        return opd_rkl_loss(RKLBatch(current_logp, sample.behavior_logp, teacher_logp)).loss
    if method == "rl":
        # Score-function form of E[log q] + alpha H(p).  At alpha=1 this
        # advantage is identical to sampled reverse-KL OPD.
        advantage = teacher_logp - alpha * sample.behavior_logp
        return rl_loss(RLBatch(current_logp, sample.behavior_logp, advantage)).loss
    raise ValueError(f"unknown trajectory method {method!r}")


def run_optimizer_trajectory(
    method: TrajectoryMethod,
    *,
    teacher: GaussianMixture1D | None = None,
    grid: QuadratureGrid1D | None = None,
    initial_mean: float = 0.0,
    initial_std: float = 2.0,
    alpha: float = 0.0,
    steps: int = 300,
    learning_rate: float = 0.03,
    sample_count: int = 2048,
    record_every: int = 10,
) -> OptimizerTrajectory:
    """Run a local path only after a caller has inspected the landscape."""

    if alpha < 0.0 or steps < 1 or learning_rate <= 0.0 or sample_count < 2 or record_every < 1:
        raise ValueError("invalid trajectory settings")
    teacher = asymmetric_teacher() if teacher is None else teacher
    grid = QuadratureGrid1D.composite(dtype=teacher.means.dtype, device=teacher.means.device) if grid is None else grid
    policy = GaussianPolicy1D(
        initial_mean,
        initial_std,
        dtype=teacher.means.dtype,
        device=teacher.means.device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    demonstration_sample = teacher.deterministic_sample(sample_count) if method == "sft" else None
    demonstrations = None if demonstration_sample is None else demonstration_sample.action
    points: list[TrajectoryPoint] = []

    def record(step: int, loss_value: Tensor) -> None:
        points.append(
            TrajectoryPoint(
                step=step,
                shared_surrogate_loss=float(loss_value.detach()),
                metrics=evaluate_policy(policy, teacher, grid),
            )
        )

    initial_loss = _shared_sampled_loss(
        policy,
        teacher,
        method,
        sample_count=sample_count,
        alpha=alpha,
        demonstrations=demonstrations,
    )
    record(0, initial_loss)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_value = _shared_sampled_loss(
            policy,
            teacher,
            method,
            sample_count=sample_count,
            alpha=alpha,
            demonstrations=demonstrations,
        )
        loss_value.backward()
        optimizer.step()
        if step % record_every == 0 or step == steps:
            post_step_loss = _shared_sampled_loss(
                policy,
                teacher,
                method,
                sample_count=sample_count,
                alpha=alpha,
                demonstrations=demonstrations,
            )
            record(step, post_step_loss)

    return OptimizerTrajectory(
        method=method,
        alpha=alpha,
        initial_mean=initial_mean,
        initial_std=initial_std,
        points=tuple(points),
    )


@torch.no_grad()
def pure_rl_collapse_scan(
    stds: Sequence[float] = (0.5, 0.2, 0.1, 0.05, 0.02),
    *,
    teacher: GaussianMixture1D | None = None,
    grid: QuadratureGrid1D | None = None,
) -> list[tuple[float, float]]:
    """Show that pure RL improves as variance shrinks at the right peak."""

    teacher = asymmetric_teacher() if teacher is None else teacher
    grid = QuadratureGrid1D.composite(dtype=teacher.means.dtype, device=teacher.means.device) if grid is None else grid
    std_tensor = torch.as_tensor(stds, dtype=teacher.means.dtype, device=teacher.means.device)
    if std_tensor.ndim != 1 or not bool(torch.all(std_tensor > 0.0)):
        raise ValueError("stds must be a one-dimensional positive sequence")
    means = torch.full_like(std_tensor, 4.0)
    log_density = normal_log_prob(grid.points[None, :], means[:, None], std_tensor.log()[:, None])
    log_mass = grid.normalize_log_density(log_density)
    utility = teacher.log_prob(grid.points).expand_as(log_mass)
    losses = rl_population_values(log_mass, utility)
    return [(float(std), float(loss)) for std, loss in zip(std_tensor, losses, strict=True)]


@dataclass(frozen=True)
class AsymmetricStudy:
    predictions: TheoryPredictions
    landscape: ObjectiveLandscape
    numerical_switch_alpha: float
    phase_rows: tuple[PhaseRow, ...]
    trajectories: tuple[OptimizerTrajectory, ...]


def run_asymmetric_study(
    *,
    landscape_config: LandscapeConfig = LandscapeConfig(),
    alphas: Sequence[float] = (0.0, 0.2, 0.35, 0.39, 0.39794, 0.41, 0.5, 0.75, 1.0),
    trajectory_steps: int = 200,
    trajectory_sample_count: int = 2048,
) -> AsymmetricStudy:
    """Run predictions, global search, and only then representative paths."""

    teacher = asymmetric_teacher()
    grid = QuadratureGrid1D.composite(dtype=teacher.means.dtype, device=teacher.means.device)
    predictions = theory_predictions(teacher)
    landscape = build_objective_landscape(teacher, grid=grid, config=landscape_config)
    switch = numerical_entropy_switch(landscape)
    phase_rows = tuple(entropy_phase_sweep(landscape, alphas))

    # Central paths show what a neutral initialization does.  Right/left paths
    # make local-basin hysteresis visible instead of hiding it in one seed.
    trajectory_specs: tuple[tuple[TrajectoryMethod, float, float, float], ...] = (
        ("sft", 0.0, 0.0, 2.0),
        ("opd_rkl", 0.0, 0.0, 2.0),
        ("opd_rkl", 0.0, 4.0, 0.1),
        ("rl", 0.0, 0.0, 2.0),
        ("rl", 0.0, -4.0, 1.0),
        ("rl", 0.35, 4.0, 0.1),
        ("rl", 0.50, -4.0, 1.0),
        ("rl", 1.0, 0.0, 2.0),
    )
    trajectories = tuple(
        run_optimizer_trajectory(
            method,
            teacher=teacher,
            grid=grid,
            initial_mean=initial_mean,
            initial_std=initial_std,
            alpha=alpha,
            steps=trajectory_steps,
            sample_count=trajectory_sample_count,
        )
        for method, alpha, initial_mean, initial_std in trajectory_specs
    )
    return AsymmetricStudy(
        predictions=predictions,
        landscape=landscape,
        numerical_switch_alpha=switch,
        phase_rows=phase_rows,
        trajectories=trajectories,
    )


__all__ = [
    "AsymmetricStudy",
    "LandscapeConfig",
    "LandscapeMinimum",
    "ObjectiveLandscape",
    "OptimizerTrajectory",
    "PhaseRow",
    "PolicyMetrics",
    "TheoryPredictions",
    "TrajectoryPoint",
    "asymmetric_teacher",
    "build_objective_landscape",
    "entropy_phase_sweep",
    "evaluate_policy",
    "isolated_mode_switch_alpha",
    "numerical_entropy_switch",
    "pure_rl_collapse_scan",
    "run_asymmetric_study",
    "run_optimizer_trajectory",
    "theory_predictions",
]
