"""Exact null control for claims that one update rule forgets less.

The student is a two-parameter binary logistic policy,

``p_theta(a=1 | x) = sigmoid(theta dot x)``.

The old context has feature ``x_old=(1, 0)`` and is initially learned exactly.
The conflicting new context has the unit feature
``x_new=(rho, sqrt(1-rho**2))``.  Consequently every new-context objective has
a rank-one gradient parallel to ``x_new``.  If an update changes the new logit
by ``delta_new``, it must change the old logit by

``delta_old = rho * delta_new``.

That identity is the preregistered prediction: at ``rho=0`` no method forgets,
and at matched new-policy progress every method following the same direction
has the same old-policy change.  A fixed learning rate does *not* make a fair
retention comparison because raw gradient magnitudes differ.

All SFT, OPD-FKL, OPD-RKL, and RL objectives, categorical metrics, and the
behavioral-KL matched step are imported from :mod:`toy_methods`.  This module
only defines the student geometry, comparison protocols, and records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, sqrt
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

try:  # Support package imports and running from this examples directory.
    from .toy_methods import (
        FKLBatch,
        categorical_metrics,
        kl_matched_step_,
        opd_fkl_loss,
        opd_rkl_population_loss,
        rl_population_loss,
        sft_population_loss,
    )
except ImportError:  # pragma: no cover - standalone example usage
    from toy_methods import (
        FKLBatch,
        categorical_metrics,
        kl_matched_step_,
        opd_fkl_loss,
        opd_rkl_population_loss,
        rl_population_loss,
        sft_population_loss,
    )


Method = Literal["sft", "opd_fkl", "opd_rkl", "rl", "rl_calibrated"]
Regime = Literal["raw_step", "matched_new_gain", "matched_behavior_kl"]

METHODS: tuple[Method, ...] = ("sft", "opd_fkl", "opd_rkl", "rl", "rl_calibrated")
REGIMES: tuple[Regime, ...] = ("raw_step", "matched_new_gain", "matched_behavior_kl")


@dataclass(frozen=True)
class RetentionConfig:
    """Shared task and comparison settings.

    Actions are ordered ``(0, 1)``.  The old teacher puts probability ``0.9``
    on action 1, while the new teacher puts probability ``0.1`` on action 1.
    Pure RL rewards action 0 and therefore has a deterministic endpoint.  The
    calibrated RL control uses ``utility=log(q_new)``, a uniform reference,
    and ``kl_beta=1``; its endpoint is exactly ``q_new``.  The uniform
    reference is only an endpoint-calibration control, not replay or an anchor
    to the old policy.
    """

    old_positive_prob: float = 0.9
    new_positive_prob: float = 0.1
    rhos: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    raw_learning_rate: float = 1.0
    target_new_gain: float = 0.02
    target_behavior_kl: float = 0.002
    initial_match_scale: float = 1e-3
    max_match_scale: float = 1e3
    match_rtol: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 < self.old_positive_prob < 1.0:
            raise ValueError("old_positive_prob must lie in (0, 1)")
        if not 0.0 < self.new_positive_prob < 1.0:
            raise ValueError("new_positive_prob must lie in (0, 1)")
        if not self.old_positive_prob > self.new_positive_prob:
            raise ValueError("the old and new targets must conflict: old_positive_prob > new_positive_prob")
        if not self.rhos:
            raise ValueError("rhos must be non-empty")
        if any(not 0.0 <= rho <= 1.0 for rho in self.rhos):
            raise ValueError("each rho must lie in [0, 1]")
        if self.raw_learning_rate <= 0.0:
            raise ValueError("raw_learning_rate must be positive")
        if self.target_new_gain <= 0.0 or self.target_behavior_kl <= 0.0:
            raise ValueError("matching targets must be positive")
        if not 0.0 < self.initial_match_scale <= self.max_match_scale:
            raise ValueError("require 0 < initial_match_scale <= max_match_scale")
        if not 0.0 < self.match_rtol < 1.0:
            raise ValueError("match_rtol must lie in (0, 1)")


@dataclass(frozen=True)
class MethodDefinition:
    """Explicit target, utility, and endpoint for one comparison label."""

    method: Method
    objective: str
    teacher_positive_prob: float | None
    utility_action_0: float | None
    utility_action_1: float | None
    reference: str | None
    kl_beta: float
    endpoint_positive_prob: float


@dataclass(frozen=True)
class RetentionRecord:
    """One exact update, with both raw and progress-normalized diagnostics."""

    regime: Regime
    method: Method
    rho: float
    loss_before: float
    raw_gradient_norm: float
    gradient_off_feature_norm: float
    step_scale: float
    step_bracketed: bool
    parameter_step_norm: float
    old_positive_before: float
    old_positive_after: float
    old_positive_drop: float
    old_target_kl_before: float
    old_target_kl_after: float
    old_target_kl_increase: float
    new_positive_before: float
    new_positive_after: float
    new_target_kl_before: float
    new_target_kl_after: float
    new_target_gain: float
    old_behavior_kl: float
    new_behavior_kl: float
    predicted_old_positive_after: float
    prediction_residual: float
    method_endpoint_positive_prob: float
    target_new_gain: float | None
    target_behavior_kl: float | None


@dataclass(frozen=True)
class RetentionStudy:
    config: RetentionConfig
    method_definitions: tuple[MethodDefinition, ...]
    records: tuple[RetentionRecord, ...]

    def select(
        self,
        *,
        regime: Regime | None = None,
        method: Method | None = None,
        rho: float | None = None,
    ) -> tuple[RetentionRecord, ...]:
        """Filter records without requiring pandas in the experiment module."""

        return tuple(
            record
            for record in self.records
            if (regime is None or record.regime == regime)
            and (method is None or record.method == method)
            and (rho is None or record.rho == rho)
        )


class BinaryLogisticStudent(nn.Module):
    """Two-parameter binary policy with action-0 logit fixed to zero."""

    def __init__(self, theta: Tensor) -> None:
        super().__init__()
        if theta.shape != (2,) or not theta.is_floating_point():
            raise ValueError("theta must be a floating tensor of shape (2,)")
        self.theta = nn.Parameter(theta.detach().clone())

    def logits(self, feature: Tensor) -> Tensor:
        if feature.shape != (2,):
            raise ValueError("feature must have shape (2,)")
        if feature.device != self.theta.device:
            raise ValueError("feature and theta must be on the same device")
        score = torch.dot(self.theta, feature.to(dtype=self.theta.dtype))
        return torch.stack((torch.zeros_like(score), score))

    def log_probs(self, feature: Tensor) -> Tensor:
        return self.logits(feature).log_softmax(dim=-1)

    def positive_probability(self, feature: Tensor) -> Tensor:
        return self.log_probs(feature)[1].exp()


@dataclass(frozen=True)
class _Baseline:
    theta: Tensor
    old_log_probs: Tensor
    new_log_probs: Tensor
    old_target_kl: float
    new_target_kl: float


@dataclass(frozen=True)
class _GainStepReport:
    scale: float
    bracketed: bool


def context_features(
    rho: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Return unit old/new features whose inner product is exactly ``rho``."""

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    old = torch.tensor((1.0, 0.0), dtype=dtype, device=device)
    new = torch.tensor((rho, sqrt(max(0.0, 1.0 - rho * rho))), dtype=dtype, device=device)
    return old, new


def target_probs(
    positive_prob: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    if not 0.0 < positive_prob < 1.0:
        raise ValueError("positive_prob must lie in (0, 1)")
    return torch.tensor((1.0 - positive_prob, positive_prob), dtype=dtype, device=device)


def initial_student(
    config: RetentionConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BinaryLogisticStudent:
    """Initialize the old context exactly at its teacher distribution."""

    old_logit = log(config.old_positive_prob / (1.0 - config.old_positive_prob))
    theta = torch.tensor((old_logit, 0.0), dtype=dtype, device=device)
    return BinaryLogisticStudent(theta)


def method_definitions(config: RetentionConfig) -> tuple[MethodDefinition, ...]:
    """Make the endpoint mismatch in the pure-RL comparison explicit."""

    q = config.new_positive_prob
    log_q0 = log(1.0 - q)
    log_q1 = log(q)
    return (
        MethodDefinition("sft", "E_q[-log p]", q, None, None, None, 0.0, q),
        MethodDefinition("opd_fkl", "E_q[-log p] = KL(q || p) + H(q)", q, None, None, None, 0.0, q),
        MethodDefinition("opd_rkl", "KL(p || q)", q, None, None, None, 0.0, q),
        MethodDefinition("rl", "-E_p[u]", None, 1.0, 0.0, None, 0.0, 0.0),
        MethodDefinition(
            "rl_calibrated",
            "-E_p[log q] + KL(p || uniform)",
            q,
            log_q0,
            log_q1,
            "uniform",
            1.0,
            q,
        ),
    )


def _new_objective(
    student: BinaryLogisticStudent,
    new_feature: Tensor,
    new_teacher: Tensor,
    method: Method,
) -> Tensor:
    """Route every objective through the shared loss implementations."""

    logits = student.logits(new_feature)
    if method == "sft":
        return sft_population_loss(logits, new_teacher)
    if method == "opd_fkl":
        return opd_fkl_loss(FKLBatch(logits.log_softmax(dim=-1), new_teacher)).loss
    if method == "opd_rkl":
        return opd_rkl_population_loss(logits, new_teacher)
    if method == "rl":
        utility = torch.tensor((1.0, 0.0), dtype=logits.dtype, device=logits.device)
        return rl_population_loss(logits, utility)
    if method == "rl_calibrated":
        reference = torch.full_like(new_teacher, 0.5)
        return rl_population_loss(
            logits,
            new_teacher.log(),
            reference_probs=reference,
            kl_beta=1.0,
        )
    raise ValueError(f"unknown method {method!r}")


def _teacher_kl(student_log_probs: Tensor, teacher_probs: Tensor) -> float:
    metrics = categorical_metrics(student_log_probs, teacher_log_probs=teacher_probs.log())
    return float(metrics["kl_teacher_student"])


def _behavior_kl(current_log_probs: Tensor, snapshot_log_probs: Tensor) -> float:
    metrics = categorical_metrics(current_log_probs, teacher_log_probs=snapshot_log_probs)
    return float(metrics["kl_teacher_student"])


def _baseline(
    student: BinaryLogisticStudent,
    old_feature: Tensor,
    new_feature: Tensor,
    old_teacher: Tensor,
    new_teacher: Tensor,
) -> _Baseline:
    old_log_probs = student.log_probs(old_feature).detach().clone()
    new_log_probs = student.log_probs(new_feature).detach().clone()
    return _Baseline(
        theta=student.theta.detach().clone(),
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        old_target_kl=_teacher_kl(old_log_probs, old_teacher),
        new_target_kl=_teacher_kl(new_log_probs, new_teacher),
    )


@torch.no_grad()
def _set_theta(student: BinaryLogisticStudent, theta: Tensor) -> None:
    student.theta.copy_(theta)


def _gain_matched_step_(
    student: BinaryLogisticStudent,
    direction: Tensor,
    new_feature: Tensor,
    new_teacher: Tensor,
    baseline: _Baseline,
    config: RetentionConfig,
) -> _GainStepReport:
    """Line-search the first point attaining a fixed new-target KL reduction."""

    if direction.shape != student.theta.shape:
        raise ValueError("direction must have the same shape as theta")

    def gain_at(scale: float) -> float:
        _set_theta(student, baseline.theta + scale * direction)
        current_kl = _teacher_kl(student.log_probs(new_feature), new_teacher)
        return baseline.new_target_kl - current_kl

    low = 0.0
    high = config.initial_match_scale
    high_gain = gain_at(high)
    previous_gain = 0.0
    while high_gain < config.target_new_gain and high < config.max_match_scale:
        if high_gain + 1e-14 < previous_gain:
            _set_theta(student, baseline.theta)
            return _GainStepReport(scale=0.0, bracketed=False)
        low = high
        previous_gain = high_gain
        high = min(2.0 * high, config.max_match_scale)
        high_gain = gain_at(high)
    if high_gain < config.target_new_gain:
        _set_theta(student, baseline.theta)
        return _GainStepReport(scale=0.0, bracketed=False)

    accepted = high
    for _ in range(60):
        mid = 0.5 * (low + high)
        mid_gain = gain_at(mid)
        accepted = mid
        if abs(mid_gain - config.target_new_gain) <= config.match_rtol * config.target_new_gain:
            break
        if mid_gain < config.target_new_gain:
            low = mid
        else:
            high = mid
    _set_theta(student, baseline.theta + accepted * direction)
    return _GainStepReport(scale=accepted, bracketed=True)


def _record(
    *,
    regime: Regime,
    method: Method,
    rho: float,
    config: RetentionConfig,
    definition: MethodDefinition,
    student: BinaryLogisticStudent,
    old_feature: Tensor,
    new_feature: Tensor,
    old_teacher: Tensor,
    new_teacher: Tensor,
    baseline: _Baseline,
    loss_before: float,
    gradient: Tensor,
    step_scale: float,
    bracketed: bool,
) -> RetentionRecord:
    old_log_probs = student.log_probs(old_feature)
    new_log_probs = student.log_probs(new_feature)
    old_target_kl = _teacher_kl(old_log_probs, old_teacher)
    new_target_kl = _teacher_kl(new_log_probs, new_teacher)
    old_positive_before = float(baseline.old_log_probs[1].exp())
    old_positive_after = float(old_log_probs[1].exp().detach())
    new_positive_before = float(baseline.new_log_probs[1].exp())
    new_positive_after = float(new_log_probs[1].exp().detach())

    projection = torch.dot(gradient, new_feature) * new_feature
    off_feature = float((gradient - projection).norm())
    # Since action 0 has fixed logit zero, log(p1/p0) is the action-1 logit.
    new_logit_before = float(baseline.new_log_probs[1] - baseline.new_log_probs[0])
    new_logit_after = float((new_log_probs[1] - new_log_probs[0]).detach())
    new_logit_delta = new_logit_after - new_logit_before
    old_logit_before = float(baseline.old_log_probs[1] - baseline.old_log_probs[0])
    predicted_logit = old_log_probs.new_tensor(old_logit_before + rho * new_logit_delta)
    predicted_old_positive = float(torch.sigmoid(predicted_logit).detach())

    return RetentionRecord(
        regime=regime,
        method=method,
        rho=rho,
        loss_before=loss_before,
        raw_gradient_norm=float(gradient.norm()),
        gradient_off_feature_norm=off_feature,
        step_scale=step_scale,
        step_bracketed=bracketed,
        parameter_step_norm=float((student.theta.detach() - baseline.theta).norm()),
        old_positive_before=old_positive_before,
        old_positive_after=old_positive_after,
        old_positive_drop=old_positive_before - old_positive_after,
        old_target_kl_before=baseline.old_target_kl,
        old_target_kl_after=old_target_kl,
        old_target_kl_increase=old_target_kl - baseline.old_target_kl,
        new_positive_before=new_positive_before,
        new_positive_after=new_positive_after,
        new_target_kl_before=baseline.new_target_kl,
        new_target_kl_after=new_target_kl,
        new_target_gain=baseline.new_target_kl - new_target_kl,
        old_behavior_kl=_behavior_kl(old_log_probs, baseline.old_log_probs),
        new_behavior_kl=_behavior_kl(new_log_probs, baseline.new_log_probs),
        predicted_old_positive_after=predicted_old_positive,
        prediction_residual=old_positive_after - predicted_old_positive,
        method_endpoint_positive_prob=definition.endpoint_positive_prob,
        target_new_gain=config.target_new_gain if regime == "matched_new_gain" else None,
        target_behavior_kl=config.target_behavior_kl if regime == "matched_behavior_kl" else None,
    )


def _run_one(
    regime: Regime,
    method: Method,
    rho: float,
    config: RetentionConfig,
    definition: MethodDefinition,
) -> RetentionRecord:
    student = initial_student(config)
    old_feature, new_feature = context_features(rho)
    old_teacher = target_probs(config.old_positive_prob)
    new_teacher = target_probs(config.new_positive_prob)
    baseline = _baseline(student, old_feature, new_feature, old_teacher, new_teacher)
    objective = _new_objective(student, new_feature, new_teacher, method)
    gradient = torch.autograd.grad(objective, student.theta)[0].detach()
    direction = -gradient

    if regime == "raw_step":
        _set_theta(student, baseline.theta + config.raw_learning_rate * direction)
        step_scale = config.raw_learning_rate
        bracketed = True
    elif regime == "matched_new_gain":
        report = _gain_matched_step_(student, direction, new_feature, new_teacher, baseline, config)
        step_scale = report.scale
        bracketed = report.bracketed
    elif regime == "matched_behavior_kl":
        # Rebuild the graph because the diagnostic gradient above consumed it.
        objective = _new_objective(student, new_feature, new_teacher, method)

        def new_context_kl_from_snapshot() -> Tensor:
            return categorical_metrics(
                student.log_probs(new_feature),
                teacher_log_probs=baseline.new_log_probs,
            )["kl_teacher_student"]

        report = kl_matched_step_(
            student,
            objective,
            new_context_kl_from_snapshot,
            target_kl=config.target_behavior_kl,
            initial_scale=config.initial_match_scale,
            max_scale=config.max_match_scale,
            rtol=config.match_rtol,
            max_iter=60,
        )
        step_scale = report.scale
        bracketed = report.bracketed
    else:  # pragma: no cover - guarded by the public literal and loop
        raise ValueError(f"unknown regime {regime!r}")

    return _record(
        regime=regime,
        method=method,
        rho=rho,
        config=config,
        definition=definition,
        student=student,
        old_feature=old_feature,
        new_feature=new_feature,
        old_teacher=old_teacher,
        new_teacher=new_teacher,
        baseline=baseline,
        loss_before=float(objective.detach()),
        gradient=gradient,
        step_scale=step_scale,
        bracketed=bracketed,
    )


def run_retention_study(
    config: RetentionConfig | None = None,
    *,
    methods: Sequence[Method] = METHODS,
    regimes: Sequence[Regime] = REGIMES,
) -> RetentionStudy:
    """Run the exact feature-overlap sweep; typical CPU runtime is a few seconds."""

    config = RetentionConfig() if config is None else config
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a non-empty subset of {METHODS}")
    if not regimes or any(regime not in REGIMES for regime in regimes):
        raise ValueError(f"regimes must be a non-empty subset of {REGIMES}")
    definitions = method_definitions(config)
    by_method = {definition.method: definition for definition in definitions}
    records = tuple(
        _run_one(regime, method, rho, config, by_method[method])
        for regime in regimes
        for rho in config.rhos
        for method in methods
    )
    return RetentionStudy(config=config, method_definitions=definitions, records=records)


def retention_rows(study: RetentionStudy, *, regime: Regime | None = None) -> list[dict[str, object]]:
    """Return full records as dictionaries for direct DataFrame construction."""

    return [asdict(record) for record in study.select(regime=regime)]


def compact_retention_rows(study: RetentionStudy, *, regime: Regime) -> list[dict[str, object]]:
    """Small notebook table focused on fairness and the overlap prediction."""

    return [
        {
            "regime": record.regime,
            "method": record.method,
            "rho": record.rho,
            "raw_gradient_norm": record.raw_gradient_norm,
            "new_positive_before": record.new_positive_before,
            "new_target_gain": record.new_target_gain,
            "new_behavior_kl": record.new_behavior_kl,
            "old_target_kl_increase": record.old_target_kl_increase,
            "old_positive_drop": record.old_positive_drop,
            "prediction_residual": record.prediction_residual,
        }
        for record in study.select(regime=regime)
    ]
