"""Transparent one-dimensional families for the theory toy experiments.

This module contains distributions and parameterizations, not training
objectives.  Every experiment calls the shared operators in ``toy_methods``.
The default dtype is float64 because several experiments compare nearly tied
population optima rather than train large neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Sequence

import torch
from torch import Tensor, nn

try:  # Support both ``python families.py``-style and package imports.
    from .toy_methods import Sample, normalized_grid_log_mass
except ImportError:  # pragma: no cover - exercised when examples are put on sys.path
    from toy_methods import Sample, normalized_grid_log_mass


DEFAULT_DTYPE = torch.float64
LOG_2PI = log(2.0 * pi)


def _as_floating_tensor(
    value: Tensor | Sequence[float] | float,
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | str = "cpu",
) -> Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if not tensor.is_floating_point():
        raise TypeError("expected a floating-point tensor")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError("values must be finite")
    return tensor


def normal_log_prob(x: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Log density of ``Normal(mean, exp(log_std)^2)`` with broadcasting."""

    if not (x.is_floating_point() and mean.is_floating_point() and log_std.is_floating_point()):
        raise TypeError("x, mean, and log_std must be floating point")
    standardized = (x - mean) * torch.exp(-log_std)
    return -0.5 * standardized.square() - log_std - 0.5 * LOG_2PI


def midpoint_normal_quantiles(
    n: int,
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Deterministic, equally weighted midpoint quantiles of ``Normal(0, 1)``.

    These points let the sampled shared OPD/RL operators be evaluated without
    Monte Carlo noise.  They are not presented as Gaussian quadrature: their
    benefit is that every point has the equal weight expected by a minibatch
    mean while the approximation converges deterministically as ``n`` grows.
    """

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    probabilities = (torch.arange(n, dtype=dtype, device=device) + 0.5) / n
    return sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)


@dataclass(frozen=True)
class QuadratureGrid1D:
    """Ordered integration points with positive Voronoi/trapezoid weights."""

    points: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        if self.points.ndim != 1 or self.points.numel() < 2:
            raise ValueError("points must be a one-dimensional tensor of length >= 2")
        if self.weights.shape != self.points.shape:
            raise ValueError("weights must have the same shape as points")
        if not (self.points.is_floating_point() and self.weights.is_floating_point()):
            raise TypeError("points and weights must be floating point")
        if not bool(torch.all(torch.isfinite(self.points))):
            raise ValueError("points must be finite")
        if not bool(torch.all(torch.diff(self.points) > 0.0)):
            raise ValueError("points must be strictly increasing")
        if not bool(torch.all(torch.isfinite(self.weights) & (self.weights > 0.0))):
            raise ValueError("weights must be finite and strictly positive")

    @classmethod
    def from_points(cls, points: Tensor) -> QuadratureGrid1D:
        """Build cell-width weights from arbitrary increasing points."""

        if points.ndim != 1 or points.numel() < 2:
            raise ValueError("points must be a one-dimensional tensor of length >= 2")
        differences = torch.diff(points)
        if not bool(torch.all(torch.isfinite(points))) or not bool(torch.all(differences > 0.0)):
            raise ValueError("points must be finite and strictly increasing")
        weights = torch.empty_like(points)
        weights[0] = 0.5 * differences[0]
        weights[-1] = 0.5 * differences[-1]
        weights[1:-1] = 0.5 * (points[2:] - points[:-2])
        return cls(points=points, weights=weights)

    @classmethod
    def composite(
        cls,
        *,
        lower: float = -30.0,
        dense_lower: float = -8.0,
        dense_upper: float = 6.0,
        upper: float = 30.0,
        tail_step: float = 0.05,
        dense_step: float = 0.005,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: torch.device | str = "cpu",
    ) -> QuadratureGrid1D:
        """A tail-covering grid that resolves the narrow component in our toy.

        The defaults place fine cells throughout all candidate policy means,
        while coarser tails prevent wide SFT projections from being silently
        renormalized on a short plotting interval.
        """

        if not lower < dense_lower < dense_upper < upper:
            raise ValueError("require lower < dense_lower < dense_upper < upper")
        if tail_step <= 0.0 or dense_step <= 0.0:
            raise ValueError("grid steps must be positive")
        left = torch.arange(lower, dense_lower, tail_step, dtype=dtype, device=device)
        dense = torch.arange(dense_lower, dense_upper, dense_step, dtype=dtype, device=device)
        right = torch.arange(dense_upper, upper + 0.5 * tail_step, tail_step, dtype=dtype, device=device)
        points = torch.unique(torch.cat((left, dense, right)), sorted=True)
        return cls.from_points(points)

    def normalize_log_density(self, log_density: Tensor) -> Tensor:
        """Return normalized log masses using the shared grid helper."""

        return normalized_grid_log_mass(log_density, dx=self.weights)


@dataclass(frozen=True)
class GaussianMixture1D:
    """A fixed finite mixture of one-dimensional Gaussian densities."""

    weights: Tensor
    means: Tensor
    stds: Tensor

    def __post_init__(self) -> None:
        if self.weights.ndim != 1 or self.weights.numel() < 1:
            raise ValueError("weights must be a non-empty one-dimensional tensor")
        if self.means.shape != self.weights.shape or self.stds.shape != self.weights.shape:
            raise ValueError("weights, means, and stds must have the same shape")
        if not (self.weights.is_floating_point() and self.means.is_floating_point() and self.stds.is_floating_point()):
            raise TypeError("mixture tensors must be floating point")
        if not (self.weights.device == self.means.device == self.stds.device):
            raise ValueError("mixture tensors must share a device")
        if not (self.weights.dtype == self.means.dtype == self.stds.dtype):
            raise ValueError("mixture tensors must share a dtype")
        if not bool(torch.all(torch.isfinite(self.weights))):
            raise ValueError("weights must be finite")
        if not bool(torch.all(torch.isfinite(self.means))):
            raise ValueError("means must be finite")
        if not bool(torch.all(torch.isfinite(self.stds) & (self.stds > 0.0))):
            raise ValueError("stds must be finite and strictly positive")
        if not bool(torch.all(self.weights > 0.0)):
            raise ValueError("weights must be strictly positive")
        if not bool(torch.isclose(self.weights.sum(), self.weights.new_tensor(1.0), atol=1e-12, rtol=1e-12)):
            raise ValueError("weights must sum to one")

    @classmethod
    def from_parameters(
        cls,
        weights: Tensor | Sequence[float],
        means: Tensor | Sequence[float],
        stds: Tensor | Sequence[float],
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: torch.device | str = "cpu",
    ) -> GaussianMixture1D:
        return cls(
            weights=_as_floating_tensor(weights, dtype=dtype, device=device),
            means=_as_floating_tensor(means, dtype=dtype, device=device),
            stds=_as_floating_tensor(stds, dtype=dtype, device=device),
        )

    @property
    def mean(self) -> Tensor:
        return torch.dot(self.weights, self.means)

    @property
    def variance(self) -> Tensor:
        centered = self.means - self.mean
        return torch.dot(self.weights, self.stds.square() + centered.square())

    def component_log_prob(self, x: Tensor) -> Tensor:
        """Component log densities, with component index on the final axis."""

        x = x.to(dtype=self.means.dtype, device=self.means.device)
        return normal_log_prob(x.unsqueeze(-1), self.means, self.stds.log())

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.logsumexp(self.weights.log() + self.component_log_prob(x), dim=-1)

    def grid_log_mass(self, grid: QuadratureGrid1D) -> Tensor:
        if grid.points.dtype != self.means.dtype or grid.points.device != self.means.device:
            raise ValueError("grid and mixture must share dtype and device")
        return grid.normalize_log_density(self.log_prob(grid.points))

    def deterministic_sample(self, n: int) -> Sample:
        """Stratified equal-weight samples, allocating counts by largest remainder."""

        if not isinstance(n, int) or isinstance(n, bool) or n < self.weights.numel():
            raise ValueError("n must be an integer at least as large as the component count")
        # Reserve one point per component, then use largest-remainder
        # allocation for the rest.  This keeps tiny components represented
        # even in a deliberately small diagnostic minibatch.
        counts = torch.ones_like(self.weights, dtype=torch.int64)
        expected = self.weights * (n - self.weights.numel())
        allocated = expected.floor().to(torch.int64)
        counts += allocated
        remaining = n - int(counts.sum())
        if remaining:
            fractional = expected - allocated
            counts[fractional.topk(remaining).indices] += 1
        actions = []
        for mean, std, count in zip(self.means, self.stds, counts, strict=True):
            z = midpoint_normal_quantiles(int(count), dtype=self.means.dtype, device=self.means.device)
            actions.append(mean + std * z)
        action = torch.cat(actions).detach()
        return Sample(action=action, behavior_logp=self.log_prob(action).detach())


class GaussianPolicy1D(nn.Module):
    """One Gaussian with directly inspectable mean and log-standard-deviation."""

    mean: nn.Parameter
    log_std: nn.Parameter

    def __init__(
        self,
        mean: float,
        std: float,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not std > 0.0:
            raise ValueError("std must be strictly positive")
        self.mean = nn.Parameter(torch.tensor(float(mean), dtype=dtype, device=device))
        self.log_std = nn.Parameter(torch.tensor(log(float(std)), dtype=dtype, device=device))

    @property
    def std(self) -> Tensor:
        return self.log_std.exp()

    @property
    def variance(self) -> Tensor:
        return torch.exp(2.0 * self.log_std)

    def log_prob(self, action: Tensor) -> Tensor:
        return normal_log_prob(action.to(self.mean), self.mean, self.log_std)

    def entropy(self) -> Tensor:
        return self.log_std + 0.5 * (1.0 + LOG_2PI)

    @torch.no_grad()
    def deterministic_sample(self, n: int) -> Sample:
        z = midpoint_normal_quantiles(n, dtype=self.mean.dtype, device=self.mean.device)
        action = self.mean + self.std * z
        return Sample(action=action.detach(), behavior_logp=self.log_prob(action).detach())

    def grid_log_mass(self, grid: QuadratureGrid1D) -> Tensor:
        if grid.points.dtype != self.mean.dtype or grid.points.device != self.mean.device:
            raise ValueError("grid and policy must share dtype and device")
        return grid.normalize_log_density(self.log_prob(grid.points))

    def probability_left_of(self, threshold: float | Tensor) -> Tensor:
        threshold_tensor = torch.as_tensor(threshold, dtype=self.mean.dtype, device=self.mean.device)
        z = (threshold_tensor - self.mean) / self.std
        return 0.5 * (1.0 + torch.erf(z / sqrt(2.0)))


__all__ = [
    "DEFAULT_DTYPE",
    "GaussianMixture1D",
    "GaussianPolicy1D",
    "QuadratureGrid1D",
    "midpoint_normal_quantiles",
    "normal_log_prob",
]
