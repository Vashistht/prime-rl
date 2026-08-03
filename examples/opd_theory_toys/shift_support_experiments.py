"""First-principles shifted-support experiments for SFT, OPD, and RL.

The student is a fixed-width Gaussian with one trainable parameter: location.
We discretize it on an ordered grid and compare two teachers whose useful mass
is shifted by the same amount:

``gaussian_tail``
    A shifted Gaussian.  Its log density changes at every student sample, so
    reverse-KL OPD has a graded direction before the student visits the target.

``flat_interval``
    A high-density target interval mixed with a flat epsilon background.  Away
    from the interval, the teacher log density is constant.  Reverse-KL OPD and
    binary-reward RL therefore only have population direction through the tiny
    student overlap with the interval (plus finite-sample score noise).

For an interval hit probability ``chi`` and ``N`` independent on-policy
samples, the preregistered discovery model is

``P(discovery) = 1 - (1 - chi)**N``, with the transition at ``N * chi ~= 1``.

This module defines distributions, sweeps, and records only.  Every SFT, OPD,
FKL, and RL loss is imported from :mod:`toy_methods`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, expm1, floor, inf, log, log1p, log2, sqrt
from typing import Literal, Sequence

import torch
from torch import Tensor

try:  # Support package imports and running from this examples directory.
    from .families import DEFAULT_DTYPE, GaussianMixture1D, GaussianPolicy1D, QuadratureGrid1D
    from .toy_methods import (
        FKLBatch,
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_kl,
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
    from families import DEFAULT_DTYPE, GaussianMixture1D, GaussianPolicy1D, QuadratureGrid1D
    from toy_methods import (
        FKLBatch,
        RKLBatch,
        RLBatch,
        SFTBatch,
        categorical_kl,
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


TeacherShape = Literal["gaussian_tail", "flat_interval", "sparse_reward"]
Method = Literal["sft", "opd_rkl", "opd_fkl", "rl"]
Estimator = Literal["population", "sampled", "full_information"]
SampleSource = Literal["none", "teacher", "student"]
SampleRegime = Literal[
    "population",
    "teacher_sample",
    "discovery_hit",
    "tail_signal_no_hit",
    "score_noise_no_hit",
    "zero_gradient_no_hit",
]


@dataclass(frozen=True)
class ShiftConfig:
    """Numerical choices shared by every method in a sweep."""

    grid_lower: float = -10.0
    grid_upper: float = 10.0
    grid_step: float = 0.05
    student_mean: float = 0.0
    student_std: float = 1.0
    teacher_std: float = 1.0
    interval_width: float = 1.0
    target_kl: float = 5e-4
    max_step_scale: float = 1e10

    def __post_init__(self) -> None:
        if not self.grid_lower < self.grid_upper:
            raise ValueError("grid_lower must be smaller than grid_upper")
        if self.grid_step <= 0.0:
            raise ValueError("grid_step must be positive")
        if self.student_std <= 0.0 or self.teacher_std <= 0.0:
            raise ValueError("student_std and teacher_std must be positive")
        if self.interval_width <= 0.0:
            raise ValueError("interval_width must be positive")
        if self.target_kl <= 0.0 or self.max_step_scale <= 0.0:
            raise ValueError("target_kl and max_step_scale must be positive")


@dataclass(frozen=True)
class SignalPrediction:
    """Infinite-domain location-family predictions fixed before optimization."""

    teacher_shape: Literal["gaussian_tail", "flat_interval"]
    displacement: float
    epsilon: float | None
    chi: float
    dchi_toward: float
    sft_toward_signal: float
    opd_toward_signal: float
    sparse_rl_toward_signal: float
    flat_log_density_contrast: float | None
    effective_batch_boundary: float


@dataclass(frozen=True)
class UpdateRecord:
    """One exact or sampled update, including information and step geometry."""

    teacher_shape: TeacherShape
    method: Method
    estimator: Estimator
    displacement: float
    epsilon: float | None
    batch_size: int | None
    seed: int | None
    sample_source: SampleSource
    sample_regime: SampleRegime
    target_hits: int | None
    chi: float
    n_chi: float | None
    predicted_discovery: float | None
    loss: float
    raw_toward_signal: float
    grad_norm: float
    step_bracketed: bool
    step_scale: float
    realized_kl: float
    mean_delta_toward: float
    target_mass_after: float


@dataclass(frozen=True)
class DiscoveryRecord:
    displacement: float
    batch_size: int
    chi: float
    n_chi: float
    predicted_probability: float
    observed_probability: float
    standard_error: float
    trials: int


@dataclass(frozen=True)
class BridgePlan:
    """Idealized support curriculum implied only by the ``N chi >= 1`` rule.

    Each stage assumes the student is successfully recentered before the next
    stage.  It is a reachability calculation, not a claim about convergence.
    """

    displacement: float
    batch_size: int
    direct_chi: float
    direct_n_chi: float
    direct_discovery_probability: float
    max_discoverable_stage_shift: float
    stage_count: int | None
    stage_centers: tuple[float, ...]


@dataclass(frozen=True)
class ShiftSweepResult:
    predictions: tuple[SignalPrediction, ...]
    exact_updates: tuple[UpdateRecord, ...]
    sampled_updates: tuple[UpdateRecord, ...]
    discovery: tuple[DiscoveryRecord, ...]
    bridge_plans: tuple[BridgePlan, ...]


def ordered_grid(
    config: ShiftConfig = ShiftConfig(),
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | str = "cpu",
) -> QuadratureGrid1D:
    """Return an inclusive, equally spaced ordered quadrature grid."""

    intervals = round((config.grid_upper - config.grid_lower) / config.grid_step)
    if intervals < 2:
        raise ValueError("grid must contain at least three points")
    reconstructed_upper = config.grid_lower + intervals * config.grid_step
    if abs(reconstructed_upper - config.grid_upper) > 1e-10 * max(1.0, abs(config.grid_upper)):
        raise ValueError("grid range must be an integer multiple of grid_step")
    points = torch.linspace(
        config.grid_lower,
        config.grid_upper,
        intervals + 1,
        dtype=dtype,
        device=device,
    )
    return QuadratureGrid1D.from_points(points)


def fixed_width_student(
    config: ShiftConfig = ShiftConfig(),
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | str = "cpu",
) -> GaussianPolicy1D:
    """Build the shared Gaussian family with only its location trainable."""

    student = GaussianPolicy1D(config.student_mean, config.student_std, dtype=dtype, device=device)
    student.log_std.requires_grad_(False)
    return student


def student_log_probs(student: GaussianPolicy1D, grid: QuadratureGrid1D) -> Tensor:
    """Normalized grid-bin log probabilities for a fixed-width student."""

    if student.log_std.requires_grad:
        raise ValueError("shift-support experiments require a frozen student width")
    return student.grid_log_mass(grid)


def target_mask(grid: QuadratureGrid1D, center: float, width: float) -> Tensor:
    if width <= 0.0:
        raise ValueError("width must be positive")
    half_width = 0.5 * width
    tolerance = 16.0 * torch.finfo(grid.points.dtype).eps * max(1.0, abs(center), abs(width))
    mask = (grid.points >= center - half_width - tolerance) & (
        grid.points <= center + half_width + tolerance
    )
    if not bool(torch.any(mask)):
        raise ValueError("target interval contains no grid points")
    return mask


def gaussian_tail_teacher(
    grid: QuadratureGrid1D,
    center: float,
    std: float,
) -> Tensor:
    """A shifted Gaussian teacher with informative tails everywhere."""

    teacher = GaussianMixture1D.from_parameters(
        weights=(1.0,),
        means=(center,),
        stds=(std,),
        dtype=grid.points.dtype,
        device=grid.points.device,
    )
    return teacher.grid_log_mass(grid).exp()


def flat_interval_teacher(
    grid: QuadratureGrid1D,
    center: float,
    width: float,
    epsilon: float,
) -> Tensor:
    """High-density interval plus an exactly flat full-grid background.

    ``epsilon`` is the *total* mixture weight of the uniform background, not a
    per-bin floor.  Thus epsilon values are comparable across grid resolution.
    """

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    mask = target_mask(grid, center, width)
    background = grid.weights / grid.weights.sum()
    interval_weights = grid.weights * mask.to(grid.weights.dtype)
    interval = interval_weights / interval_weights.sum()
    teacher = (1.0 - epsilon) * interval + epsilon * background
    return teacher / teacher.sum()


def interval_probability(log_probs: Tensor, mask: Tensor) -> Tensor:
    if log_probs.shape != mask.shape:
        raise ValueError("log_probs and target mask must have the same shape")
    return log_probs.exp()[mask].sum()


def discovery_probability(chi: float, batch_size: int) -> float:
    """Stable evaluation of ``1 - (1 - chi)**batch_size``."""

    if not 0.0 <= chi <= 1.0:
        raise ValueError("chi must lie in [0, 1]")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if chi == 1.0:
        return 1.0
    return -expm1(batch_size * log1p(-chi))


def effective_batch_boundary(chi: float) -> float:
    """The batch size at which the effective count ``N chi`` equals one."""

    if not 0.0 <= chi <= 1.0:
        raise ValueError("chi must lie in [0, 1]")
    return inf if chi == 0.0 else 1.0 / chi


def bracketing_batch_sizes(chi: float) -> tuple[int, int]:
    """Power-of-two batches below and above the ``N chi = 1`` boundary.

    The lower batch is at most half of ``1/chi``; the upper batch is the first
    power of two at or above it.  This avoids a wasteful Cartesian product of
    every batch size with shifts whose discovery boundaries differ by orders
    of magnitude.
    """

    boundary = effective_batch_boundary(chi)
    if boundary == inf:
        raise ValueError("cannot bracket a zero-probability event")
    lower = max(1, 2 ** floor(log2(max(1.0, boundary / 2.0))))
    upper = max(2 * lower, 2 ** ceil(log2(max(1.0, boundary))))
    return lower, upper


def _toward_sign(displacement: float) -> float:
    if displacement == 0.0:
        raise ValueError("displacement must be nonzero")
    return 1.0 if displacement > 0.0 else -1.0


def _location_score(student: GaussianPolicy1D, log_probs: Tensor, grid: QuadratureGrid1D) -> Tensor:
    """Derivative of normalized grid log mass with respect to location."""

    mean_on_grid = torch.dot(log_probs.exp(), grid.points)
    return (grid.points - mean_on_grid) / student.variance.detach()


@torch.no_grad()
def signal_predictions(
    displacement: float,
    epsilon: float,
    *,
    config: ShiftConfig = ShiftConfig(),
    grid: QuadratureGrid1D | None = None,
) -> tuple[SignalPrediction, SignalPrediction]:
    """Preregister location-signal scalings for the two teacher shapes.

    The formulas are exact for an untruncated fixed-width Gaussian location
    family.  Grid truncation errors are negligible when the configured domain
    covers the student and teacher tails.
    """

    sign = _toward_sign(displacement)
    grid = ordered_grid(config) if grid is None else grid
    center = config.student_mean + displacement
    mask = target_mask(grid, center, config.interval_width)
    student = fixed_width_student(config, dtype=grid.points.dtype, device=grid.points.device)
    log_probs = student_log_probs(student, grid)
    probabilities = log_probs.exp()
    score = _location_score(student, log_probs, grid)
    chi_tensor = probabilities[mask].sum()
    dchi_dmean = torch.dot(probabilities * mask.to(probabilities.dtype), score)
    dchi_toward = sign * float(dchi_dmean)
    chi = float(chi_tensor)

    gaussian = gaussian_tail_teacher(grid, center, config.teacher_std)
    flat = flat_interval_teacher(grid, center, config.interval_width, epsilon)
    student_grid_mean = float(torch.dot(probabilities, grid.points))
    gaussian_mean = float(torch.dot(gaussian, grid.points))
    flat_mean = float(torch.dot(flat, grid.points))

    domain_measure = float(grid.weights.sum())
    interval_measure = float(grid.weights[mask].sum())
    outside_density = epsilon / domain_measure
    inside_density = (1.0 - epsilon) / interval_measure + outside_density
    contrast = log(inside_density / outside_density)
    boundary = effective_batch_boundary(chi)
    common_rl = dchi_toward

    return (
        SignalPrediction(
            teacher_shape="gaussian_tail",
            displacement=displacement,
            epsilon=None,
            chi=chi,
            dchi_toward=dchi_toward,
            sft_toward_signal=sign * (gaussian_mean - student_grid_mean) / config.student_std**2,
            opd_toward_signal=abs(displacement) / config.teacher_std**2,
            sparse_rl_toward_signal=common_rl,
            flat_log_density_contrast=None,
            effective_batch_boundary=boundary,
        ),
        SignalPrediction(
            teacher_shape="flat_interval",
            displacement=displacement,
            epsilon=epsilon,
            chi=chi,
            dchi_toward=dchi_toward,
            sft_toward_signal=sign * (flat_mean - student_grid_mean) / config.student_std**2,
            opd_toward_signal=contrast * dchi_toward,
            sparse_rl_toward_signal=common_rl,
            flat_log_density_contrast=contrast,
            effective_batch_boundary=boundary,
        ),
    )


def _fresh_student(config: ShiftConfig, grid: QuadratureGrid1D) -> GaussianPolicy1D:
    return fixed_width_student(config, dtype=grid.points.dtype, device=grid.points.device)


def _record_update(
    *,
    student: GaussianPolicy1D,
    loss_value: Tensor,
    grid: QuadratureGrid1D,
    mask: Tensor,
    config: ShiftConfig,
    teacher_shape: TeacherShape,
    method: Method,
    estimator: Estimator,
    displacement: float,
    epsilon: float | None,
    batch_size: int | None,
    seed: int | None,
    sample_source: SampleSource,
    target_hits: int | None,
) -> UpdateRecord:
    old_log_probs = student_log_probs(student, grid).detach()
    old_probabilities = old_log_probs.exp()
    chi = float(old_probabilities[mask].sum())
    sign = _toward_sign(displacement)
    gradient = torch.autograd.grad(loss_value, student.mean, retain_graph=True)[0]
    raw_toward_signal = -sign * float(gradient.detach())
    gradient_magnitude = abs(float(gradient.detach()))

    if sample_source == "none":
        sample_regime: SampleRegime = "population"
    elif sample_source == "teacher":
        sample_regime = "teacher_sample"
    elif target_hits is not None and target_hits > 0:
        sample_regime = "discovery_hit"
    elif method == "opd_rkl" and teacher_shape == "gaussian_tail":
        sample_regime = "tail_signal_no_hit"
    elif gradient_magnitude == 0.0:
        sample_regime = "zero_gradient_no_hit"
    else:
        # For flat q, a no-hit minibatch contains no directional teacher
        # information.  Its nonzero empirical score sum is estimator noise.
        sample_regime = "score_noise_no_hit"

    def old_to_current_kl() -> Tensor:
        return categorical_kl(old_probabilities, student_log_probs(student, grid).exp())

    step = kl_matched_step_(
        student,
        loss_value,
        old_to_current_kl,
        target_kl=config.target_kl,
        initial_scale=1.0,
        max_scale=config.max_step_scale,
        rtol=2e-4,
    )
    mean_delta_toward = sign * (float(student.mean.detach()) - config.student_mean)
    target_mass_after = float(interval_probability(student_log_probs(student, grid), mask).detach())
    predicted = None if batch_size is None or sample_source != "student" else discovery_probability(chi, batch_size)
    return UpdateRecord(
        teacher_shape=teacher_shape,
        method=method,
        estimator=estimator,
        displacement=displacement,
        epsilon=epsilon,
        batch_size=batch_size,
        seed=seed,
        sample_source=sample_source,
        sample_regime=sample_regime,
        target_hits=target_hits,
        chi=chi,
        n_chi=None if batch_size is None else batch_size * chi,
        predicted_discovery=predicted,
        loss=float(loss_value.detach()),
        raw_toward_signal=raw_toward_signal,
        grad_norm=step.grad_norm,
        step_bracketed=step.bracketed,
        step_scale=step.scale,
        realized_kl=step.realized_kl,
        mean_delta_toward=mean_delta_toward,
        target_mass_after=target_mass_after,
    )


def _population_update(
    *,
    method: Method,
    teacher_shape: TeacherShape,
    teacher_probs: Tensor | None,
    reward: Tensor,
    displacement: float,
    epsilon: float | None,
    config: ShiftConfig,
    grid: QuadratureGrid1D,
    mask: Tensor,
) -> UpdateRecord:
    student = _fresh_student(config, grid)
    log_probs = student_log_probs(student, grid)
    if method == "sft":
        if teacher_probs is None:
            raise ValueError("SFT requires teacher_probs")
        loss_value = sft_population_loss(log_probs, teacher_probs)
        estimator: Estimator = "population"
    elif method == "opd_rkl":
        if teacher_probs is None:
            raise ValueError("OPD-RKL requires teacher_probs")
        loss_value = opd_rkl_population_loss(log_probs, teacher_probs)
        estimator = "population"
    elif method == "opd_fkl":
        if teacher_probs is None:
            raise ValueError("OPD-FKL requires teacher_probs")
        loss_value = opd_fkl_loss(FKLBatch(log_probs, teacher_probs)).loss
        estimator = "full_information"
    elif method == "rl":
        loss_value = rl_population_loss(log_probs, reward)
        estimator = "population"
    else:  # pragma: no cover - protected by Literal and internal callers
        raise ValueError(f"unknown method {method!r}")
    return _record_update(
        student=student,
        loss_value=loss_value,
        grid=grid,
        mask=mask,
        config=config,
        teacher_shape=teacher_shape,
        method=method,
        estimator=estimator,
        displacement=displacement,
        epsilon=epsilon,
        batch_size=None,
        seed=None,
        sample_source="none",
        target_hits=None,
    )


def _sampled_update(
    *,
    method: Literal["sft", "opd_rkl", "rl"],
    teacher_shape: TeacherShape,
    teacher_probs: Tensor | None,
    reward: Tensor,
    displacement: float,
    epsilon: float | None,
    batch_size: int,
    seed: int,
    config: ShiftConfig,
    grid: QuadratureGrid1D,
    mask: Tensor,
) -> UpdateRecord:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    student = _fresh_student(config, grid)
    log_probs = student_log_probs(student, grid)
    generator = torch.Generator(device=grid.points.device).manual_seed(seed)

    if method == "sft":
        if teacher_probs is None:
            raise ValueError("sampled SFT requires teacher_probs")
        sample = sample_categorical(teacher_probs.log(), batch_size, generator=generator)
        loss_value = sft_loss(SFTBatch(log_probs[sample.action])).loss
        source: SampleSource = "teacher"
    else:
        # Reinitializing the generator for each method gives OPD and RL common
        # random numbers: identical student actions at a fixed (N, seed).
        sample = sample_categorical(log_probs.detach(), batch_size, generator=generator)
        current_logp = log_probs[sample.action]
        if method == "opd_rkl":
            if teacher_probs is None:
                raise ValueError("sampled OPD-RKL requires teacher_probs")
            loss_value = opd_rkl_loss(
                RKLBatch(current_logp, sample.behavior_logp, teacher_probs.log()[sample.action])
            ).loss
        elif method == "rl":
            # Uncentered binary reward makes a nonzero update equivalent to at
            # least one discovery, exactly matching 1-(1-chi)^N.  A baseline
            # changes variance but not the population gradient.
            loss_value = rl_loss(RLBatch(current_logp, sample.behavior_logp, reward[sample.action])).loss
        else:  # pragma: no cover
            raise ValueError(f"unknown sampled method {method!r}")
        source = "student"

    hits = int(mask[sample.action].sum())
    return _record_update(
        student=student,
        loss_value=loss_value,
        grid=grid,
        mask=mask,
        config=config,
        teacher_shape=teacher_shape,
        method=method,
        estimator="sampled",
        displacement=displacement,
        epsilon=epsilon,
        batch_size=batch_size,
        seed=seed,
        sample_source=source,
        target_hits=hits,
    )


@torch.no_grad()
def verify_discovery_model(
    displacement: float,
    batch_sizes: Sequence[int],
    seeds: Sequence[int],
    *,
    config: ShiftConfig = ShiftConfig(),
    grid: QuadratureGrid1D | None = None,
) -> tuple[DiscoveryRecord, ...]:
    """Verify the preregistered discovery law with seeded on-policy samples."""

    _toward_sign(displacement)
    if not batch_sizes or not seeds:
        raise ValueError("batch_sizes and seeds must be non-empty")
    if any((not isinstance(n, int) or isinstance(n, bool) or n < 1) for n in batch_sizes):
        raise ValueError("all batch_sizes must be positive integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique independent trials")
    grid = ordered_grid(config) if grid is None else grid
    center = config.student_mean + displacement
    mask = target_mask(grid, center, config.interval_width)
    student = _fresh_student(config, grid)
    log_probs = student_log_probs(student, grid)
    chi = float(interval_probability(log_probs, mask))
    maximum_n = max(batch_sizes)
    discoveries = {n: 0 for n in batch_sizes}
    for seed in seeds:
        generator = torch.Generator(device=grid.points.device).manual_seed(seed)
        actions = sample_categorical(log_probs, maximum_n, generator=generator).action
        cumulative_hits = mask[actions].to(torch.int64).cumsum(dim=-1)
        for n in batch_sizes:
            discoveries[n] += int(cumulative_hits[n - 1] > 0)

    records = []
    trials = len(seeds)
    for n in batch_sizes:
        predicted = discovery_probability(chi, n)
        standard_error = sqrt(predicted * (1.0 - predicted) / trials)
        records.append(
            DiscoveryRecord(
                displacement=displacement,
                batch_size=n,
                chi=chi,
                n_chi=n * chi,
                predicted_probability=predicted,
                observed_probability=discoveries[n] / trials,
                standard_error=standard_error,
                trials=trials,
            )
        )
    return tuple(records)


@torch.no_grad()
def idealized_bridge_plan(
    displacement: float,
    batch_size: int,
    *,
    config: ShiftConfig = ShiftConfig(),
    grid: QuadratureGrid1D | None = None,
) -> BridgePlan:
    """Greedy support curriculum using only the preregistered ``N chi`` rule."""

    sign = _toward_sign(displacement)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    grid = ordered_grid(config) if grid is None else grid
    student = _fresh_student(config, grid)
    log_probs = student_log_probs(student, grid)
    distance = abs(displacement)
    target_center = config.student_mean + displacement
    direct_mask = target_mask(grid, target_center, config.interval_width)
    direct_chi = float(interval_probability(log_probs, direct_mask))

    step_count = max(1, int(ceil(distance / config.grid_step)))
    candidate_shifts = torch.linspace(0.0, distance, step_count + 1, dtype=grid.points.dtype)
    feasible: list[float] = []
    for shift in candidate_shifts.tolist():
        center = config.student_mean + sign * shift
        candidate_mask = target_mask(grid, center, config.interval_width)
        chi = float(interval_probability(log_probs, candidate_mask))
        if batch_size * chi >= 1.0:
            feasible.append(shift)
    maximum_shift = max(feasible, default=0.0)

    if maximum_shift == 0.0:
        stage_count: int | None = None
        centers: tuple[float, ...] = ()
    else:
        stage_count = max(1, int(ceil(distance / maximum_shift)))
        centers = tuple(
            config.student_mean + sign * min(stage * maximum_shift, distance)
            for stage in range(1, stage_count + 1)
        )
    return BridgePlan(
        displacement=displacement,
        batch_size=batch_size,
        direct_chi=direct_chi,
        direct_n_chi=batch_size * direct_chi,
        direct_discovery_probability=discovery_probability(direct_chi, batch_size),
        max_discoverable_stage_shift=maximum_shift,
        stage_count=stage_count,
        stage_centers=centers,
    )


def run_shift_sweep(
    *,
    displacements: Sequence[float] = (2.5, 3.5, 4.5),
    epsilons: Sequence[float] = (1e-1, 1e-3, 1e-6),
    batch_sizes: Sequence[int] | None = None,
    seeds: Sequence[int] = tuple(range(8)),
    config: ShiftConfig = ShiftConfig(),
    include_bridges: bool = True,
) -> ShiftSweepResult:
    """Run exact calculations first, then common-random-number Monte Carlo.

    Gaussian-teacher results do not depend on epsilon and are therefore stored
    once per displacement.  Sparse RL depends only on the target interval and
    is likewise stored once rather than duplicated for both teachers.
    """

    if not displacements or not epsilons or not seeds:
        raise ValueError("displacements, epsilons, and seeds must be non-empty")
    if any(d == 0.0 for d in displacements):
        raise ValueError("all displacements must be nonzero")
    if any(not 0.0 < epsilon < 1.0 for epsilon in epsilons):
        raise ValueError("all epsilons must lie strictly between zero and one")
    if batch_sizes is not None and not batch_sizes:
        raise ValueError("batch_sizes must be non-empty when provided")
    if batch_sizes is not None and any(
        (not isinstance(n, int) or isinstance(n, bool) or n < 1) for n in batch_sizes
    ):
        raise ValueError("all batch_sizes must be positive integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    grid = ordered_grid(config)
    prediction_records: list[SignalPrediction] = []
    exact_records: list[UpdateRecord] = []
    teacher_sets: dict[float, list[tuple[TeacherShape, float | None, Tensor]]] = {}
    masks: dict[float, Tensor] = {}
    batch_sets: dict[float, tuple[int, ...]] = {}

    # Phase 1: all deterministic predictions and exact population updates.
    for displacement in displacements:
        center = config.student_mean + displacement
        if not config.grid_lower < center - 0.5 * config.interval_width:
            raise ValueError("target interval must lie inside the grid")
        if not center + 0.5 * config.interval_width < config.grid_upper:
            raise ValueError("target interval must lie inside the grid")
        mask = target_mask(grid, center, config.interval_width)
        masks[displacement] = mask
        reward = mask.to(grid.points.dtype)
        gaussian = gaussian_tail_teacher(grid, center, config.teacher_std)
        teacher_sets[displacement] = [("gaussian_tail", None, gaussian)]
        gaussian_prediction = signal_predictions(displacement, epsilons[0], config=config, grid=grid)[0]
        prediction_records.append(gaussian_prediction)
        batch_sets[displacement] = (
            bracketing_batch_sizes(gaussian_prediction.chi)
            if batch_sizes is None
            else tuple(batch_sizes)
        )
        for epsilon in epsilons:
            flat = flat_interval_teacher(grid, center, config.interval_width, epsilon)
            teacher_sets[displacement].append(("flat_interval", epsilon, flat))
            prediction_records.append(signal_predictions(displacement, epsilon, config=config, grid=grid)[1])

        for teacher_shape, epsilon, teacher in teacher_sets[displacement]:
            for method in ("sft", "opd_rkl", "opd_fkl"):
                exact_records.append(
                    _population_update(
                        method=method,
                        teacher_shape=teacher_shape,
                        teacher_probs=teacher,
                        reward=reward,
                        displacement=displacement,
                        epsilon=epsilon,
                        config=config,
                        grid=grid,
                        mask=mask,
                    )
                )
        exact_records.append(
            _population_update(
                method="rl",
                teacher_shape="sparse_reward",
                teacher_probs=None,
                reward=reward,
                displacement=displacement,
                epsilon=None,
                config=config,
                grid=grid,
                mask=mask,
            )
        )

    # Phase 2: sampled estimators.  No Monte Carlo result can alter phase 1.
    sampled_records: list[UpdateRecord] = []
    for displacement in displacements:
        mask = masks[displacement]
        reward = mask.to(grid.points.dtype)
        for batch_size in batch_sets[displacement]:
            for seed in seeds:
                for teacher_shape, epsilon, teacher in teacher_sets[displacement]:
                    for method in ("sft", "opd_rkl"):
                        sampled_records.append(
                            _sampled_update(
                                method=method,
                                teacher_shape=teacher_shape,
                                teacher_probs=teacher,
                                reward=reward,
                                displacement=displacement,
                                epsilon=epsilon,
                                batch_size=batch_size,
                                seed=seed,
                                config=config,
                                grid=grid,
                                mask=mask,
                            )
                        )
                sampled_records.append(
                    _sampled_update(
                        method="rl",
                        teacher_shape="sparse_reward",
                        teacher_probs=None,
                        reward=reward,
                        displacement=displacement,
                        epsilon=None,
                        batch_size=batch_size,
                        seed=seed,
                        config=config,
                        grid=grid,
                        mask=mask,
                    )
                )

    discovery_records = tuple(
        record
        for displacement in displacements
        for record in verify_discovery_model(
            displacement,
            batch_sets[displacement],
            seeds,
            config=config,
            grid=grid,
        )
    )
    bridge_records = (
        tuple(
            idealized_bridge_plan(displacement, n, config=config, grid=grid)
            for displacement in displacements
            for n in batch_sets[displacement]
        )
        if include_bridges
        else ()
    )
    return ShiftSweepResult(
        predictions=tuple(prediction_records),
        exact_updates=tuple(exact_records),
        sampled_updates=tuple(sampled_records),
        discovery=discovery_records,
        bridge_plans=bridge_records,
    )


def compact_update_rows(result: ShiftSweepResult) -> tuple[dict[str, object], ...]:
    """Aggregate seed-level updates into notebook-ready tabular rows.

    The function deliberately has no pandas dependency: pass its result to
    ``pandas.DataFrame`` if desired.  ``sample_regimes`` makes a no-hit zero RL
    gradient visibly different from flat-teacher score noise and informative
    Gaussian-tail OPD.
    """

    rows: list[dict[str, object]] = []
    for record in result.exact_updates:
        rows.append(
            {
                "teacher": record.teacher_shape,
                "method": record.method,
                "estimator": record.estimator,
                "displacement": record.displacement,
                "epsilon": record.epsilon,
                "N": None,
                "seeds": 0,
                "chi": record.chi,
                "N_chi": None,
                "p_discovery": None,
                "hit_rate": None,
                "toward_rate": float(record.raw_toward_signal > 0.0),
                "mean_raw_toward_signal": record.raw_toward_signal,
                "matched_rate": float(record.step_bracketed),
                "mean_delta_toward": record.mean_delta_toward,
                "sample_regimes": "population:1",
            }
        )

    grouped: dict[tuple[object, ...], list[UpdateRecord]] = {}
    for record in result.sampled_updates:
        key = (
            record.teacher_shape,
            record.method,
            record.estimator,
            record.displacement,
            record.epsilon,
            record.batch_size,
        )
        grouped.setdefault(key, []).append(record)
    for key, records in grouped.items():
        teacher, method, estimator, displacement, epsilon, batch_size = key
        count = len(records)
        regimes: dict[str, int] = {}
        for record in records:
            regimes[record.sample_regime] = regimes.get(record.sample_regime, 0) + 1
        rows.append(
            {
                "teacher": teacher,
                "method": method,
                "estimator": estimator,
                "displacement": displacement,
                "epsilon": epsilon,
                "N": batch_size,
                "seeds": count,
                "chi": records[0].chi,
                "N_chi": records[0].n_chi,
                "p_discovery": records[0].predicted_discovery,
                "hit_rate": sum(bool(record.target_hits) for record in records) / count,
                "toward_rate": sum(record.raw_toward_signal > 0.0 for record in records) / count,
                "mean_raw_toward_signal": sum(record.raw_toward_signal for record in records) / count,
                "matched_rate": sum(record.step_bracketed for record in records) / count,
                "mean_delta_toward": sum(record.mean_delta_toward for record in records) / count,
                "sample_regimes": ";".join(f"{name}:{regimes[name]}" for name in sorted(regimes)),
            }
        )
    return tuple(rows)


__all__ = [
    "BridgePlan",
    "DiscoveryRecord",
    "ShiftConfig",
    "ShiftSweepResult",
    "SignalPrediction",
    "UpdateRecord",
    "bracketing_batch_sizes",
    "compact_update_rows",
    "discovery_probability",
    "effective_batch_boundary",
    "fixed_width_student",
    "flat_interval_teacher",
    "gaussian_tail_teacher",
    "idealized_bridge_plan",
    "interval_probability",
    "ordered_grid",
    "run_shift_sweep",
    "signal_predictions",
    "student_log_probs",
    "target_mask",
    "verify_discovery_model",
]
