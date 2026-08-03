"""Finite-sampling companion for the truth--teacher--student benchmark.

Named modes are treated as one categorical outcome.  This deliberately strips
away location optimization so that each comparison measures only the sampling
information available to SFT, reverse-KL policy distillation, and RL.

The caller supplies three strictly positive distributions:

``truth``
    The target distribution ``p_star`` used only by truth-derived rewards and
    evaluation.  Pure RL uses reward ``log p_star(a)``.  Entropy-calibrated RL
    uses the fresh-policy score reward
    ``log p_star(a) - log p_behavior(a)`` and therefore estimates the gradient
    of ``KL(p_student || p_star)``.
``teacher``
    The distribution imitated by SFT/FKL/RKL.  A tiny positive epsilon can
    model a practically omitted teacher mode without undefined log scores.
``student``
    The initial behavior policy.  Its small masses determine on-policy mode
    discovery through ``1 - (1 - p_i)**N``.

Every objective and sampler comes from :mod:`toy_methods`; this module only
constructs batches, records gradient statistics, and evaluates discovery.
The constant-validity reward is intentionally left uncentered: its population
gradient is zero, finite-batch gradients are pure score noise, and subtracting
the batch-constant baseline would make every draw exactly zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import expm1, inf, log1p, sqrt
from typing import Literal, Sequence

import torch
from torch import Tensor

try:  # Support package imports and running from this examples directory.
    from .toy_methods import (
        FKLBatch,
        RKLBatch,
        RLBatch,
        SFTBatch,
        gradient_cosine,
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
        gradient_cosine,
        opd_fkl_loss,
        opd_rkl_loss,
        opd_rkl_population_loss,
        rl_loss,
        rl_population_loss,
        sample_categorical,
        sft_loss,
        sft_population_loss,
    )


Method = Literal[
    "sft_hard",
    "opd_fkl_full",
    "opd_rkl",
    "rl_pure_truth",
    "rl_entropy_truth",
    "rl_constant_validity",
]
SampleSource = Literal["teacher", "student"]

METHODS: tuple[Method, ...] = (
    "sft_hard",
    "opd_fkl_full",
    "opd_rkl",
    "rl_pure_truth",
    "rl_entropy_truth",
    "rl_constant_validity",
)


@dataclass(frozen=True)
class SamplingConfig:
    batch_sizes: tuple[int, ...] = (8, 32, 128, 512)
    seeds: tuple[int, ...] = tuple(range(64))

    def __post_init__(self) -> None:
        if not self.batch_sizes or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in self.batch_sizes
        ):
            raise ValueError("batch_sizes must contain positive integers")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must be unique")
        if not self.seeds or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")


@dataclass(frozen=True)
class ProbabilityVectors:
    truth: tuple[float, ...]
    teacher: tuple[float, ...]
    student: tuple[float, ...]


@dataclass(frozen=True)
class ExactGradientRecord:
    method: Method
    objective_value: float
    gradient: tuple[float, ...]
    gradient_norm: float


@dataclass(frozen=True)
class GradientDraw:
    method: Method
    batch_size: int
    seed: int
    loss: float
    gradient: tuple[float, ...]
    gradient_norm: float
    cosine_to_exact: float | None


@dataclass(frozen=True)
class GradientSummary:
    """Monte Carlo statistics against the shared population gradient.

    ``noise_rms`` is ``sqrt(E[||g - E[g]||^2])`` and
    ``snr = ||g_exact|| / noise_rms``.  ``rmse_to_exact`` additionally includes
    finite-seed bias.  Cosines are undefined when the exact gradient is zero.
    """

    method: Method
    batch_size: int
    trials: int
    exact_gradient: tuple[float, ...]
    mean_gradient: tuple[float, ...]
    std_gradient: tuple[float, ...]
    bias: tuple[float, ...]
    exact_gradient_norm: float
    mean_gradient_norm: float
    bias_norm: float
    noise_rms: float
    standard_error_norm: float
    rmse_to_exact: float
    snr: float | None
    cosine_of_mean_to_exact: float | None
    mean_draw_cosine_to_exact: float | None
    mean_loss: float
    std_loss: float


@dataclass(frozen=True)
class ModeDiscoveryRecord:
    source: SampleSource
    batch_size: int
    mode: int
    mode_probability: float
    expected_count: float
    predicted_hit_probability: float
    observed_mean_count: float
    observed_hit_rate: float
    predicted_hit_rate_standard_error: float
    trials: int


@dataclass(frozen=True)
class TruthSamplingStudy:
    probabilities: ProbabilityVectors
    config: SamplingConfig
    exact_gradients: tuple[ExactGradientRecord, ...]
    draws: tuple[GradientDraw, ...]
    summaries: tuple[GradientSummary, ...]
    discovery: tuple[ModeDiscoveryRecord, ...]

    def exact(self, method: Method) -> ExactGradientRecord:
        matches = tuple(record for record in self.exact_gradients if record.method == method)
        if len(matches) != 1:
            raise KeyError(f"expected one exact record for {method!r}, found {len(matches)}")
        return matches[0]

    def summary(self, method: Method, batch_size: int) -> GradientSummary:
        matches = tuple(
            record for record in self.summaries if record.method == method and record.batch_size == batch_size
        )
        if len(matches) != 1:
            raise KeyError(f"expected one summary for {(method, batch_size)!r}, found {len(matches)}")
        return matches[0]


def _probability_tensor(name: str, values: Sequence[float] | Tensor) -> Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach().clone()
    if result.ndim != 1 or result.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two modes")
    if bool(torch.any(~torch.isfinite(result))) or bool(torch.any(result <= 0.0)):
        raise ValueError(f"{name} probabilities must be finite and strictly positive")
    if not torch.allclose(result.sum(), torch.tensor(1.0, dtype=result.dtype), atol=1e-8, rtol=1e-8):
        raise ValueError(f"{name} probabilities must sum to one")
    return result / result.sum()


def _validated_vectors(
    truth: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    student: Sequence[float] | Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    truth_tensor = _probability_tensor("truth", truth)
    teacher_tensor = _probability_tensor("teacher", teacher)
    student_tensor = _probability_tensor("student", student)
    if not (truth_tensor.shape == teacher_tensor.shape == student_tensor.shape):
        raise ValueError("truth, teacher, and student must have the same number of modes")
    return truth_tensor, teacher_tensor, student_tensor


def _fresh_logits(student: Tensor) -> Tensor:
    return student.log().detach().clone().requires_grad_(True)


def _population_loss(method: Method, logits: Tensor, truth: Tensor, teacher: Tensor) -> Tensor:
    if method == "sft_hard":
        return sft_population_loss(logits, teacher)
    if method == "opd_fkl_full":
        return opd_fkl_loss(FKLBatch(logits.log_softmax(dim=-1), teacher)).loss
    if method == "opd_rkl":
        return opd_rkl_population_loss(logits, teacher)
    if method == "rl_pure_truth":
        return rl_population_loss(logits, truth.log())
    if method == "rl_entropy_truth":
        return rl_population_loss(logits, truth.log(), entropy_bonus=1.0)
    if method == "rl_constant_validity":
        return rl_population_loss(logits, torch.ones_like(truth))
    raise ValueError(f"unknown method {method!r}")


def _loss_gradient(loss: Tensor, logits: Tensor) -> Tensor:
    return torch.autograd.grad(loss, logits, allow_unused=False)[0].detach()


def _exact_records(truth: Tensor, teacher: Tensor, student: Tensor) -> tuple[ExactGradientRecord, ...]:
    records = []
    for method in METHODS:
        logits = _fresh_logits(student)
        loss = _population_loss(method, logits, truth, teacher)
        gradient = _loss_gradient(loss, logits)
        records.append(
            ExactGradientRecord(
                method=method,
                objective_value=float(loss.detach()),
                gradient=tuple(float(value) for value in gradient),
                gradient_norm=float(gradient.norm()),
            )
        )
    return tuple(records)


def _sampled_loss(
    method: Method,
    logits: Tensor,
    truth: Tensor,
    teacher: Tensor,
    teacher_actions: Tensor,
    student_actions: Tensor,
    student_behavior_logp: Tensor,
) -> Tensor:
    log_probs = logits.log_softmax(dim=-1)
    if method == "sft_hard":
        return sft_loss(SFTBatch(log_probs.gather(dim=-1, index=teacher_actions))).loss
    if method == "opd_fkl_full":
        return opd_fkl_loss(FKLBatch(log_probs, teacher)).loss

    student_logp = log_probs.gather(dim=-1, index=student_actions)
    if method == "opd_rkl":
        return opd_rkl_loss(
            RKLBatch(
                student_logp=student_logp,
                behavior_logp=student_behavior_logp,
                teacher_logp=teacher.log().gather(dim=-1, index=student_actions),
            )
        ).loss
    if method == "rl_pure_truth":
        advantage = truth.log().gather(dim=-1, index=student_actions)
    elif method == "rl_entropy_truth":
        advantage = truth.log().gather(dim=-1, index=student_actions) - student_behavior_logp
    elif method == "rl_constant_validity":
        advantage = torch.ones_like(student_behavior_logp)
    else:  # pragma: no cover - METHODS controls all public calls
        raise ValueError(f"unknown method {method!r}")
    return rl_loss(
        RLBatch(
            student_logp=student_logp,
            behavior_logp=student_behavior_logp,
            advantage=advantage,
        )
    ).loss


def _cosine_or_none(gradient: Tensor, exact: Tensor, *, eps: float = 1e-14) -> float | None:
    if float(exact.norm()) <= eps or float(gradient.norm()) <= eps:
        return None
    return float(gradient_cosine(gradient, exact))


def _draws_and_counts(
    truth: Tensor,
    teacher: Tensor,
    student: Tensor,
    config: SamplingConfig,
    exact_by_method: dict[Method, Tensor],
) -> tuple[tuple[GradientDraw, ...], dict[tuple[SampleSource, int], Tensor]]:
    draws: list[GradientDraw] = []
    counts: dict[tuple[SampleSource, int], list[Tensor]] = {
        (source, size): [] for source in ("teacher", "student") for size in config.batch_sizes
    }
    teacher_logits = teacher.log()
    student_logits = student.log()

    for batch_size in config.batch_sizes:
        for seed in config.seeds:
            teacher_generator = torch.Generator(device="cpu").manual_seed(seed)
            student_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000_007)
            teacher_sample = sample_categorical(teacher_logits, batch_size, generator=teacher_generator)
            student_sample = sample_categorical(student_logits, batch_size, generator=student_generator)
            counts[("teacher", batch_size)].append(
                torch.bincount(teacher_sample.action, minlength=teacher.numel()).to(torch.float64)
            )
            counts[("student", batch_size)].append(
                torch.bincount(student_sample.action, minlength=student.numel()).to(torch.float64)
            )

            for method in METHODS:
                logits = _fresh_logits(student)
                loss = _sampled_loss(
                    method,
                    logits,
                    truth,
                    teacher,
                    teacher_sample.action,
                    student_sample.action,
                    student_sample.behavior_logp,
                )
                gradient = _loss_gradient(loss, logits)
                draws.append(
                    GradientDraw(
                        method=method,
                        batch_size=batch_size,
                        seed=seed,
                        loss=float(loss.detach()),
                        gradient=tuple(float(value) for value in gradient),
                        gradient_norm=float(gradient.norm()),
                        cosine_to_exact=_cosine_or_none(gradient, exact_by_method[method]),
                    )
                )

    stacked_counts = {key: torch.stack(value) for key, value in counts.items()}
    return tuple(draws), stacked_counts


def _summaries(
    draws: tuple[GradientDraw, ...],
    exact_by_method: dict[Method, Tensor],
    config: SamplingConfig,
) -> tuple[GradientSummary, ...]:
    summaries = []
    for batch_size in config.batch_sizes:
        for method in METHODS:
            selected = tuple(record for record in draws if record.batch_size == batch_size and record.method == method)
            gradients = torch.tensor([record.gradient for record in selected], dtype=torch.float64)
            losses = torch.tensor([record.loss for record in selected], dtype=torch.float64)
            exact = exact_by_method[method]
            mean = gradients.mean(dim=0)
            std = gradients.std(dim=0, correction=0)
            bias = mean - exact
            centered = gradients - mean
            errors = gradients - exact
            noise_rms = float(centered.square().sum(dim=-1).mean().sqrt())
            if noise_rms < 1e-15:
                noise_rms = 0.0
                std = torch.zeros_like(std)
            rmse_to_exact = float(errors.square().sum(dim=-1).mean().sqrt())
            exact_norm = float(exact.norm())
            if exact_norm <= 1e-14:
                snr = 0.0 if noise_rms > 0.0 else None
            else:
                snr = inf if noise_rms == 0.0 else exact_norm / noise_rms
            cosines = [record.cosine_to_exact for record in selected if record.cosine_to_exact is not None]
            loss_std = float(losses.std(correction=0))
            if loss_std < 1e-15:
                loss_std = 0.0
            summaries.append(
                GradientSummary(
                    method=method,
                    batch_size=batch_size,
                    trials=len(selected),
                    exact_gradient=tuple(float(value) for value in exact),
                    mean_gradient=tuple(float(value) for value in mean),
                    std_gradient=tuple(float(value) for value in std),
                    bias=tuple(float(value) for value in bias),
                    exact_gradient_norm=exact_norm,
                    mean_gradient_norm=float(mean.norm()),
                    bias_norm=float(bias.norm()),
                    noise_rms=noise_rms,
                    standard_error_norm=noise_rms / sqrt(len(selected)),
                    rmse_to_exact=rmse_to_exact,
                    snr=snr,
                    cosine_of_mean_to_exact=_cosine_or_none(mean, exact),
                    mean_draw_cosine_to_exact=sum(cosines) / len(cosines) if cosines else None,
                    mean_loss=float(losses.mean()),
                    std_loss=loss_std,
                )
            )
    return tuple(summaries)


def discovery_probability(probability: float, batch_size: int) -> float:
    """Stable ``1 - (1-p)**N`` for one mode and an IID batch."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return -expm1(batch_size * log1p(-probability))


def _discovery_records(
    teacher: Tensor,
    student: Tensor,
    counts: dict[tuple[SampleSource, int], Tensor],
    config: SamplingConfig,
) -> tuple[ModeDiscoveryRecord, ...]:
    records = []
    for source, probabilities in (("teacher", teacher), ("student", student)):
        for batch_size in config.batch_sizes:
            source_counts = counts[(source, batch_size)]
            for mode, probability_tensor in enumerate(probabilities):
                probability = float(probability_tensor)
                predicted = discovery_probability(probability, batch_size)
                mode_counts = source_counts[:, mode]
                records.append(
                    ModeDiscoveryRecord(
                        source=source,
                        batch_size=batch_size,
                        mode=mode,
                        mode_probability=probability,
                        expected_count=batch_size * probability,
                        predicted_hit_probability=predicted,
                        observed_mean_count=float(mode_counts.mean()),
                        observed_hit_rate=float((mode_counts > 0).to(torch.float64).mean()),
                        predicted_hit_rate_standard_error=sqrt(predicted * (1.0 - predicted) / len(config.seeds)),
                        trials=len(config.seeds),
                    )
                )
    return tuple(records)


def run_truth_sampling_study(
    truth: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    student: Sequence[float] | Tensor,
    *,
    config: SamplingConfig | None = None,
) -> TruthSamplingStudy:
    """Compare exact and sampled initial gradients on explicit mode vectors."""

    config = SamplingConfig() if config is None else config
    truth_tensor, teacher_tensor, student_tensor = _validated_vectors(truth, teacher, student)
    exact = _exact_records(truth_tensor, teacher_tensor, student_tensor)
    exact_by_method = {record.method: torch.tensor(record.gradient, dtype=torch.float64) for record in exact}
    draws, counts = _draws_and_counts(
        truth_tensor,
        teacher_tensor,
        student_tensor,
        config,
        exact_by_method,
    )
    summaries = _summaries(draws, exact_by_method, config)
    discovery = _discovery_records(teacher_tensor, student_tensor, counts, config)
    probabilities = ProbabilityVectors(
        truth=tuple(float(value) for value in truth_tensor),
        teacher=tuple(float(value) for value in teacher_tensor),
        student=tuple(float(value) for value in student_tensor),
    )
    return TruthSamplingStudy(
        probabilities=probabilities,
        config=config,
        exact_gradients=exact,
        draws=draws,
        summaries=summaries,
        discovery=discovery,
    )


def exact_rows(study: TruthSamplingStudy) -> list[dict[str, object]]:
    return [asdict(record) for record in study.exact_gradients]


def summary_rows(study: TruthSamplingStudy) -> list[dict[str, object]]:
    return [asdict(record) for record in study.summaries]


def discovery_rows(study: TruthSamplingStudy) -> list[dict[str, object]]:
    return [asdict(record) for record in study.discovery]
