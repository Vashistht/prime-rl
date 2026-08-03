"""Reward choice and GRPO-style sampling on the named truth modes.

The truth-mixture benchmark previously used the dense oracle reward
``log p_star(a)``.  That is useful because entropy regularization with
coefficient one recovers ``p_star``, but it is not a binary verifier.  This
module keeps three reward models separate:

``oracle_log_density``
    ``R(t, a) = log p_star(a)``.  The sampled reference ``t`` is irrelevant.
``binary_reference``
    ``R(t, a) = 1[a = t]``.
``squared_distance_reference``
    ``R(t, a) = -((x_a - x_t) / spacing)^2``.

Rows index a reference ``t ~ p_star`` and columns index a student action.
Consequently, the ordinary population-RL utility is ``u = p_star @ R``.
Unregularized RL selects an argmax of ``u``; entropy-regularized RL has the
closed-form endpoint ``softmax(u / alpha)``.

The sampling study uses the shared :mod:`toy_methods` sampler, unnormalized
group centering, and RL surrogate.  One reference is drawn per group and all
``G`` actions in that group share it.  At a fresh policy, the expected update
is exactly ``(G - 1) / G`` times the ordinary policy-gradient update.  A group
is informative precisely when it crosses reward-equivalence classes.  This
matters for distance rewards: actions equidistant from the reference are a
single class, even though they are different actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import expm1, inf, isfinite, log1p, sqrt
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from torch import Tensor

try:  # Support package imports and direct execution from this directory.
    from .toy_methods import RLBatch, group_advantages, rl_loss, rl_population_loss, sample_categorical
except ImportError:  # pragma: no cover - standalone example usage
    from toy_methods import RLBatch, group_advantages, rl_loss, rl_population_loss, sample_categorical

if TYPE_CHECKING:
    try:
        from .truth_mixture_experiments import ModeMixtureProblem
    except ImportError:  # pragma: no cover - standalone type checking
        from truth_mixture_experiments import ModeMixtureProblem


RewardKind = Literal[
    "oracle_log_density",
    "binary_reference",
    "squared_distance_reference",
]
REWARD_KINDS: tuple[RewardKind, ...] = (
    "oracle_log_density",
    "binary_reference",
    "squared_distance_reference",
)


def _probability_tensor(name: str, values: Sequence[float] | Tensor) -> Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach().clone()
    if result.ndim != 1 or result.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two modes")
    if bool(torch.any(~torch.isfinite(result))) or bool(torch.any(result <= 0.0)):
        raise ValueError(f"{name} probabilities must be finite and strictly positive")
    one = torch.ones((), dtype=result.dtype)
    if not torch.allclose(result.sum(), one, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} probabilities must sum to one")
    return result / result.sum()


def _coordinate_tensor(values: Sequence[float] | Tensor, mode_count: int) -> Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach().clone()
    if result.shape != (mode_count,):
        raise ValueError(f"coordinates must be a length-{mode_count} vector")
    if bool(torch.any(~torch.isfinite(result))):
        raise ValueError("coordinates must be finite")
    if bool(torch.any(result.diff() <= 0.0)):
        raise ValueError("coordinates must be strictly increasing")
    return result


def _spacing(coordinates: Tensor, spacing: float | None) -> float:
    differences = coordinates.diff()
    if spacing is None:
        if not torch.allclose(
            differences,
            differences[0].expand_as(differences),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("spacing must be supplied when coordinates are not equally spaced")
        spacing = float(differences[0])
    if not isinstance(spacing, (float, int)) or isinstance(spacing, bool):
        raise TypeError("spacing must be a finite positive number")
    spacing = float(spacing)
    if not isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be a finite positive number")
    return spacing


def _equivalence_classes(row: Tensor) -> tuple[tuple[int, ...], ...]:
    """Partition actions by exactly equal reward, preserving action order."""

    groups: dict[float, list[int]] = {}
    for action, value in enumerate(row):
        groups.setdefault(float(value), []).append(action)
    return tuple(tuple(actions) for actions in groups.values())


@dataclass(frozen=True)
class RewardLandscape:
    """Exact reward matrix, marginalized utility, and objective endpoints."""

    kind: RewardKind
    truth_probs: Tensor
    coordinates: Tensor
    spacing: float
    matrix: Tensor
    expected_utility: Tensor
    maximizing_modes: tuple[int, ...]
    pure_endpoint: Tensor
    entropy_alpha: float
    entropy_endpoint: Tensor
    equivalence_classes: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        if self.kind not in REWARD_KINDS:
            raise ValueError(f"unknown reward kind {self.kind!r}")
        n = self.truth_probs.numel()
        if n < 2 or self.truth_probs.shape != (n,) or self.coordinates.shape != (n,):
            raise ValueError("truth probabilities and coordinates must be matching vectors")
        if self.matrix.shape != (n, n) or self.expected_utility.shape != (n,):
            raise ValueError("reward matrix and expected utility have inconsistent shapes")
        if self.pure_endpoint.shape != (n,) or self.entropy_endpoint.shape != (n,):
            raise ValueError("reward endpoints must match the number of modes")
        tensors = (
            self.truth_probs,
            self.coordinates,
            self.matrix,
            self.expected_utility,
            self.pure_endpoint,
            self.entropy_endpoint,
        )
        if any(tensor.dtype != torch.float64 or tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("reward landscape tensors must use CPU float64")
        if any(bool(torch.any(~torch.isfinite(tensor))) for tensor in tensors):
            raise ValueError("reward landscape tensors must be finite")
        if bool(torch.any(self.truth_probs <= 0.0)):
            raise ValueError("truth probabilities must be strictly positive")
        one = torch.ones((), dtype=torch.float64)
        for name, endpoint in (
            ("truth probabilities", self.truth_probs),
            ("pure endpoint", self.pure_endpoint),
            ("entropy endpoint", self.entropy_endpoint),
        ):
            if bool(torch.any(endpoint < 0.0)) or not torch.allclose(
                endpoint.sum(), one, atol=1e-10, rtol=1e-10
            ):
                raise ValueError(f"{name} must be a probability vector")
        if self.coordinates.numel() > 1 and bool(torch.any(self.coordinates.diff() <= 0.0)):
            raise ValueError("coordinates must be strictly increasing")
        if not isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("spacing must be finite and positive")
        if not isfinite(self.entropy_alpha) or self.entropy_alpha <= 0.0:
            raise ValueError("entropy_alpha must be finite and positive")
        if not torch.allclose(
            self.expected_utility,
            self.truth_probs @ self.matrix,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("expected utility must equal truth_probs @ matrix")
        if not self.maximizing_modes or len(set(self.maximizing_modes)) != len(self.maximizing_modes):
            raise ValueError("maximizing_modes must be non-empty and unique")
        if any(index < 0 or index >= n for index in self.maximizing_modes):
            raise ValueError("maximizing mode index is out of range")
        expected_pure = torch.zeros_like(self.pure_endpoint)
        expected_pure[list(self.maximizing_modes)] = 1.0 / len(self.maximizing_modes)
        if not torch.allclose(self.pure_endpoint, expected_pure, atol=1e-12, rtol=1e-12):
            raise ValueError("pure endpoint must be uniform over maximizing modes")
        expected_entropy = (self.expected_utility / self.entropy_alpha).softmax(dim=-1)
        if not torch.allclose(self.entropy_endpoint, expected_entropy, atol=1e-12, rtol=1e-12):
            raise ValueError("entropy endpoint must equal softmax(expected_utility / entropy_alpha)")
        if len(self.equivalence_classes) != n:
            raise ValueError("equivalence classes must contain one partition per reference")
        for reference, classes in enumerate(self.equivalence_classes):
            flattened = tuple(action for group in classes for action in group)
            if sorted(flattened) != list(range(n)) or len(flattened) != n:
                raise ValueError("each equivalence-class row must partition all actions")
            for group in classes:
                if not group:
                    raise ValueError("reward equivalence classes must be non-empty")
                rewards = self.matrix[reference, list(group)]
                if not bool(torch.all(rewards == rewards[0])):
                    raise ValueError("an equivalence class may only contain equal rewards")


def build_reward_landscape(
    truth_probs: Sequence[float] | Tensor,
    coordinates: Sequence[float] | Tensor,
    *,
    kind: RewardKind,
    spacing: float | None = None,
    entropy_alpha: float = 1.0,
) -> RewardLandscape:
    """Construct one reward model on a fixed truth distribution and geometry."""

    if kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward kind {kind!r}")
    if not isinstance(entropy_alpha, (float, int)) or isinstance(entropy_alpha, bool):
        raise TypeError("entropy_alpha must be a finite positive number")
    entropy_alpha = float(entropy_alpha)
    if not isfinite(entropy_alpha) or entropy_alpha <= 0.0:
        raise ValueError("entropy_alpha must be a finite positive number")
    truth = _probability_tensor("truth", truth_probs)
    coordinate_tensor = _coordinate_tensor(coordinates, truth.numel())
    nominal_spacing = _spacing(coordinate_tensor, spacing)
    n = truth.numel()

    if kind == "oracle_log_density":
        matrix = truth.log().expand(n, n).clone()
    elif kind == "binary_reference":
        matrix = torch.eye(n, dtype=torch.float64)
    else:
        normalized_displacement = (
            coordinate_tensor.unsqueeze(0) - coordinate_tensor.unsqueeze(1)
        ) / nominal_spacing
        matrix = -normalized_displacement.square()

    expected_utility = truth @ matrix
    maximum = expected_utility.max()
    tie_tolerance = 1e-12 * max(1.0, float(expected_utility.abs().max()))
    maximizing_modes = tuple(
        int(index)
        for index in torch.nonzero((expected_utility - maximum).abs() <= tie_tolerance).flatten()
    )
    pure_endpoint = torch.zeros_like(truth)
    pure_endpoint[list(maximizing_modes)] = 1.0 / len(maximizing_modes)
    entropy_endpoint = (expected_utility / entropy_alpha).softmax(dim=-1)
    classes = tuple(_equivalence_classes(matrix[reference]) for reference in range(n))
    return RewardLandscape(
        kind=kind,
        truth_probs=truth,
        coordinates=coordinate_tensor,
        spacing=nominal_spacing,
        matrix=matrix,
        expected_utility=expected_utility,
        maximizing_modes=maximizing_modes,
        pure_endpoint=pure_endpoint,
        entropy_alpha=entropy_alpha,
        entropy_endpoint=entropy_endpoint,
        equivalence_classes=classes,
    )


def build_mixture_reward_landscape(
    problem: ModeMixtureProblem,
    *,
    kind: RewardKind,
    entropy_alpha: float = 1.0,
) -> RewardLandscape:
    """Use the truth weights and coordinates from ``ModeMixtureProblem``.

    The denominator remains the nominal spacing of the unshifted mode grid,
    while the rewards use the actual truth-mode coordinates.  Thus optional
    component shifts change distance rewards without silently changing their
    unit of measurement.
    """

    base_differences = problem.base_mode_means.detach().cpu().to(torch.float64).diff()
    if base_differences.numel() < 1 or not torch.allclose(
        base_differences,
        base_differences[0].expand_as(base_differences),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("problem base-mode coordinates must be an equally spaced grid")
    return build_reward_landscape(
        problem.truth_weights,
        problem.truth_mode_means,
        kind=kind,
        spacing=float(base_differences[0]),
        entropy_alpha=entropy_alpha,
    )


def _validate_group_size(group_size: int) -> None:
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("group_size must be a positive integer")


def _validated_current_probs(landscape: RewardLandscape, values: Sequence[float] | Tensor) -> Tensor:
    current = _probability_tensor("current student", values)
    if current.shape != landscape.truth_probs.shape:
        raise ValueError("current student and reward landscape must have the same number of modes")
    return current


def exact_raw_update(landscape: RewardLandscape, current_probs: Sequence[float] | Tensor) -> Tensor:
    """Return the ordinary fresh-policy RL update ``-grad(loss)`` in logits."""

    current = _validated_current_probs(landscape, current_probs)
    logits = current.log().detach().clone().requires_grad_(True)
    loss_value = rl_population_loss(logits, landscape.expected_utility)
    return -torch.autograd.grad(loss_value, logits)[0].detach()


def expected_centered_update(
    landscape: RewardLandscape,
    current_probs: Sequence[float] | Tensor,
    group_size: int,
) -> Tensor:
    """Exact expectation of the unnormalized group-centered RL estimator."""

    _validate_group_size(group_size)
    return ((group_size - 1.0) / group_size) * exact_raw_update(landscape, current_probs)


def conditional_informative_group_probability(
    landscape: RewardLandscape,
    current_probs: Sequence[float] | Tensor,
    *,
    reference: int,
    group_size: int,
) -> float:
    """Probability that one fixed-reference group contains reward contrast."""

    _validate_group_size(group_size)
    current = _validated_current_probs(landscape, current_probs)
    if not isinstance(reference, int) or isinstance(reference, bool) or not 0 <= reference < current.numel():
        raise ValueError("reference must be a valid integer mode index")
    homogeneous = current.new_zeros(())
    for reward_class in landscape.equivalence_classes[reference]:
        class_mass = current[list(reward_class)].sum()
        homogeneous = homogeneous + class_mass.pow(group_size)
    return min(1.0, max(0.0, float(1.0 - homogeneous)))


def informative_group_probability(
    landscape: RewardLandscape,
    current_probs: Sequence[float] | Tensor,
    group_size: int,
) -> float:
    """Marginal group-contrast probability for ``t ~ p_star``."""

    _validate_group_size(group_size)
    current = _validated_current_probs(landscape, current_probs)
    conditional = current.new_tensor(
        [
            conditional_informative_group_probability(
                landscape,
                current,
                reference=reference,
                group_size=group_size,
            )
            for reference in range(current.numel())
        ]
    )
    return min(1.0, max(0.0, float(torch.dot(landscape.truth_probs, conditional))))


def informative_batch_probability(group_probability: float, group_count: int) -> float:
    """Probability at least one of ``group_count`` independent groups informs."""

    if not isinstance(group_count, int) or isinstance(group_count, bool) or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    if not isfinite(group_probability) or not 0.0 <= group_probability <= 1.0:
        raise ValueError("group_probability must lie in [0, 1]")
    if group_probability == 0.0:
        return 0.0
    if group_probability == 1.0:
        return 1.0
    return -expm1(group_count * log1p(-group_probability))


@dataclass(frozen=True)
class GroupSamplingConfig:
    """Fixed-completion budgets for a sweep over GRPO group size."""

    group_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    total_completions: int = 128
    batches_per_seed: int = 256
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    no_op_tolerance: float = 1e-14

    def __post_init__(self) -> None:
        if not self.group_sizes or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in self.group_sizes
        ):
            raise ValueError("group_sizes must contain positive integers")
        if len(set(self.group_sizes)) != len(self.group_sizes):
            raise ValueError("group_sizes must be unique")
        integers = (self.total_completions, self.batches_per_seed)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers):
            raise ValueError("total_completions and batches_per_seed must be positive integers")
        if any(self.total_completions % size != 0 for size in self.group_sizes):
            raise ValueError("every group size must divide total_completions")
        if not self.seeds or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not isfinite(self.no_op_tolerance) or self.no_op_tolerance < 0.0:
            raise ValueError("no_op_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class GroupSamplingSummary:
    """Monte Carlo update and reward-contrast statistics for one group size."""

    reward_kind: RewardKind
    group_size: int
    groups_per_batch: int
    total_completions: int
    batches: int
    exact_raw_update: tuple[float, ...]
    expected_centered_update: tuple[float, ...]
    mean_update: tuple[float, ...]
    raw_update_norm: float
    expected_update_norm: float
    mean_update_norm: float
    bias_norm: float
    noise_rms: float
    standard_error_norm: float
    rmse_to_expected: float
    snr: float | None
    cosine_of_mean_to_expected: float | None
    predicted_informative_group_probability: float
    observed_informative_group_probability: float
    predicted_informative_batch_probability: float
    observed_informative_batch_probability: float
    predicted_no_op_rate: float
    observed_no_op_rate: float


@dataclass(frozen=True)
class GroupSamplingStudy:
    landscape: RewardLandscape
    current_probs: tuple[float, ...]
    config: GroupSamplingConfig
    summaries: tuple[GroupSamplingSummary, ...]

    def summary(self, group_size: int) -> GroupSamplingSummary:
        matches = tuple(record for record in self.summaries if record.group_size == group_size)
        if len(matches) != 1:
            raise KeyError(f"expected one summary for group size {group_size}, found {len(matches)}")
        return matches[0]

    def rows(self) -> list[dict[str, object]]:
        return [asdict(summary) for summary in self.summaries]


def _cosine_or_none(first: Tensor, second: Tensor, *, eps: float = 1e-14) -> float | None:
    denominator = float(first.norm() * second.norm())
    if denominator <= eps:
        return None
    return float(torch.dot(first, second) / denominator)


def _seeded_batch_updates(
    landscape: RewardLandscape,
    current: Tensor,
    *,
    group_size: int,
    groups_per_batch: int,
    batches: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorize independent batches while evaluating the shared RL loss once."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    references = sample_categorical(
        landscape.truth_probs.log().expand(batches, -1),
        groups_per_batch,
        generator=generator,
    ).action
    behavior = sample_categorical(
        current.log().reshape(1, 1, -1).expand(batches, groups_per_batch, -1),
        group_size,
        generator=generator,
    )
    rewards = landscape.matrix[references.unsqueeze(-1), behavior.action]
    advantages = group_advantages(rewards, normalize=False)

    batch_logits = current.log().expand(batches, -1).clone().requires_grad_(True)
    student_rows = batch_logits.log_softmax(dim=-1).unsqueeze(1).expand(-1, groups_per_batch, -1)
    student_logp = student_rows.gather(dim=-1, index=behavior.action)
    loss_value = rl_loss(
        RLBatch(
            student_logp=student_logp,
            behavior_logp=behavior.behavior_logp,
            advantage=advantages,
        )
    ).loss
    # rl_loss averages over every batch as well as over each batch's
    # completions.  Each logit row is independent, so multiplying by the
    # number of rows recovers one update estimator per batch.
    updates = -batches * torch.autograd.grad(loss_value, batch_logits)[0].detach()
    informative_groups = (rewards != rewards[..., :1]).any(dim=-1)
    informative_batches = informative_groups.any(dim=-1)
    return updates, informative_groups, informative_batches


def run_group_sampling_study(
    landscape: RewardLandscape,
    current_probs: Sequence[float] | Tensor,
    *,
    config: GroupSamplingConfig | None = None,
) -> GroupSamplingStudy:
    """Sweep group size at fixed completions using Prime-style centering."""

    config = GroupSamplingConfig() if config is None else config
    if not isinstance(config, GroupSamplingConfig):
        raise TypeError("config must be a GroupSamplingConfig")
    current = _validated_current_probs(landscape, current_probs)
    raw_update = exact_raw_update(landscape, current)
    summaries: list[GroupSamplingSummary] = []

    for group_size in config.group_sizes:
        groups_per_batch = config.total_completions // group_size
        update_parts: list[Tensor] = []
        group_parts: list[Tensor] = []
        batch_parts: list[Tensor] = []
        for seed in config.seeds:
            updates, informative_groups, informative_batches = _seeded_batch_updates(
                landscape,
                current,
                group_size=group_size,
                groups_per_batch=groups_per_batch,
                batches=config.batches_per_seed,
                seed=seed,
            )
            update_parts.append(updates)
            group_parts.append(informative_groups)
            batch_parts.append(informative_batches)

        all_updates = torch.cat(update_parts)
        all_informative_groups = torch.cat(group_parts)
        all_informative_batches = torch.cat(batch_parts)
        expected = ((group_size - 1.0) / group_size) * raw_update
        mean = all_updates.mean(dim=0)
        centered = all_updates - mean
        errors = all_updates - expected
        noise_rms = float(centered.square().sum(dim=-1).mean().sqrt())
        if noise_rms < 1e-15:
            noise_rms = 0.0
        expected_norm = float(expected.norm())
        if expected_norm <= 1e-14:
            snr = 0.0 if noise_rms > 0.0 else None
        else:
            snr = inf if noise_rms == 0.0 else expected_norm / noise_rms
        predicted_group = informative_group_probability(landscape, current, group_size)
        predicted_batch = informative_batch_probability(predicted_group, groups_per_batch)
        no_op = all_updates.norm(dim=-1) <= config.no_op_tolerance
        summaries.append(
            GroupSamplingSummary(
                reward_kind=landscape.kind,
                group_size=group_size,
                groups_per_batch=groups_per_batch,
                total_completions=config.total_completions,
                batches=all_updates.shape[0],
                exact_raw_update=tuple(float(value) for value in raw_update),
                expected_centered_update=tuple(float(value) for value in expected),
                mean_update=tuple(float(value) for value in mean),
                raw_update_norm=float(raw_update.norm()),
                expected_update_norm=expected_norm,
                mean_update_norm=float(mean.norm()),
                bias_norm=float((mean - expected).norm()),
                noise_rms=noise_rms,
                standard_error_norm=noise_rms / sqrt(all_updates.shape[0]),
                rmse_to_expected=float(errors.square().sum(dim=-1).mean().sqrt()),
                snr=snr,
                cosine_of_mean_to_expected=_cosine_or_none(mean, expected),
                predicted_informative_group_probability=predicted_group,
                observed_informative_group_probability=float(all_informative_groups.to(torch.float64).mean()),
                predicted_informative_batch_probability=predicted_batch,
                observed_informative_batch_probability=float(all_informative_batches.to(torch.float64).mean()),
                predicted_no_op_rate=1.0 - predicted_batch,
                observed_no_op_rate=float(no_op.to(torch.float64).mean()),
            )
        )

    return GroupSamplingStudy(
        landscape=landscape,
        current_probs=tuple(float(value) for value in current),
        config=config,
        summaries=tuple(summaries),
    )


def summary_rows(study: GroupSamplingStudy) -> list[dict[str, object]]:
    """Notebook-ready records for a reward/group-size study."""

    return study.rows()
