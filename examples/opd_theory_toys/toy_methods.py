"""Shared toy SFT, OPD, and RL operators.

This is the only file in the toy study that implements the basic update
operators.  Experiment modules specify distributions and collect data, then
call these functions.  All losses are means over valid observations so that
batch size and padding do not silently change an update's scale.

Conventions
-----------
``student_logp`` is always recomputed under the trainable policy.
``behavior_logp`` is always detached data from the policy that sampled an
action.  Losses are minimized.  Thus ``-gradient(loss)`` is the update
direction discussed in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor, nn

Method = Literal["sft", "opd_rkl", "opd_fkl", "rl"]


@dataclass(frozen=True)
class SFTBatch:
    """Hard demonstration actions evaluated under the current student."""

    student_logp_demo: Tensor
    mask: Tensor | None = None


@dataclass(frozen=True)
class RKLBatch:
    """Student-sampled actions with teacher and behavior log probabilities."""

    student_logp: Tensor
    behavior_logp: Tensor
    teacher_logp: Tensor
    mask: Tensor | None = None


@dataclass(frozen=True)
class FKLBatch:
    """Full teacher probabilities and student log probabilities per state."""

    student_log_probs: Tensor
    teacher_probs: Tensor
    mask: Tensor | None = None


@dataclass(frozen=True)
class RLBatch:
    """Student-sampled actions with detached scalar advantages."""

    student_logp: Tensor
    behavior_logp: Tensor
    advantage: Tensor
    mask: Tensor | None = None


@dataclass(frozen=True)
class LossReport:
    loss: Tensor
    metrics: dict[str, Tensor]


@dataclass(frozen=True)
class Sample:
    """A sampled action/value and its detached behavior log probability."""

    action: Tensor
    behavior_logp: Tensor


# Compatibility for the first categorical-only version of this module.
CategoricalSample = Sample


@dataclass(frozen=True)
class StepReport:
    loss: float
    grad_norm: float
    scale: float
    realized_kl: float
    target_kl: float
    bracketed: bool


@dataclass(frozen=True)
class FixedStepReport:
    """Report for a common-scale gradient step with an optional norm cap."""

    loss: float
    grad_norm: float
    requested_scale: float
    applied_scale: float
    max_grad_norm: float | None
    clipped: bool
    realized_kl: float
    no_op: bool


def _validated_mask(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != value shape {tuple(values.shape)}")
    return mask.to(device=values.device, dtype=torch.bool)


def masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean over valid elements; an all-padding batch contributes zero.

    Avoiding Python conversion of ``count`` keeps this reducer compatible with
    ``torch.vmap`` and ``torch.compile``.  An all-false mask retains a valid
    zero gradient through ``values``.
    """

    valid = _validated_mask(values, mask)
    count = valid.sum()
    return (values * valid.to(values.dtype)).sum() / count.clamp_min(1)


def weighted_mean(values: Tensor, weights: Tensor, mask: Tensor | None = None) -> Tensor:
    """Normalize a non-negative weighted reduction over valid elements.

    ``weights`` are treated as fixed sampling/reweighting data.  An empty mask
    or all-zero effective weights returns zero while retaining a valid zero
    gradient through ``values``.  Requiring an exact shape match keeps state
    occupancy weights from broadcasting across an unintended dimension.
    """

    if weights.shape != values.shape:
        raise ValueError(f"weight shape {tuple(weights.shape)} != value shape {tuple(values.shape)}")
    if not weights.is_floating_point():
        raise TypeError("weights must be floating point")
    if weights.device != values.device:
        raise ValueError("weights and values must be on the same device")
    fixed_weights = weights.detach().to(dtype=values.dtype)
    if bool(torch.any(~torch.isfinite(fixed_weights))) or bool(torch.any(fixed_weights < 0)):
        raise ValueError("weights must be finite and non-negative")
    valid = _validated_mask(values, mask)
    effective = torch.where(valid, fixed_weights, torch.zeros_like(fixed_weights))
    numerator = torch.where(effective > 0, values * effective, torch.zeros_like(values)).sum()
    denominator = effective.sum()
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return numerator / safe_denominator


def sft_loss(batch: SFTBatch) -> LossReport:
    """Hard-label SFT: ``mean[-log p_theta(a_demo | s_demo)]``."""

    nll = -batch.student_logp_demo
    value = masked_mean(nll, batch.mask)
    return LossReport(loss=value, metrics={"nll": value.detach()})


def opd_rkl_loss(batch: RKLBatch, *, mismatch_penalty: float = 0.0) -> LossReport:
    """Prime-like sampled reverse-KL OPD surrogate.

    For actions sampled from behavior ``b``, the loss is

    ``- (p_theta(a)/b(a)) stopgrad[log q(a)-log p_theta(a)]``.

    At a fresh policy (``b=p``), its expected gradient is the gradient of
    ``KL(p || q)``.  ``mismatch_penalty`` optionally adds the same local
    squared-log-ratio trust penalty used by Prime-RL, without clipping/masking.
    """

    if mismatch_penalty < 0:
        raise ValueError("mismatch_penalty must be non-negative")
    if not (batch.student_logp.shape == batch.behavior_logp.shape == batch.teacher_logp.shape):
        raise ValueError("student, behavior, and teacher log probabilities must have the same shape")
    behavior = batch.behavior_logp.detach()
    teacher = batch.teacher_logp.detach()
    log_ratio = batch.student_logp - behavior
    importance_ratio = log_ratio.exp()
    teacher_advantage = (teacher - batch.student_logp).detach()
    per_item = -teacher_advantage * importance_ratio + mismatch_penalty * log_ratio.square()
    value = masked_mean(per_item, batch.mask)
    valid = _validated_mask(per_item, batch.mask)
    metrics = {
        "teacher_advantage": masked_mean(teacher_advantage, valid).detach(),
        "importance_ratio": masked_mean(importance_ratio, valid).detach(),
        "log_ratio_sq": masked_mean(log_ratio.square(), valid).detach(),
    }
    return LossReport(loss=value, metrics=metrics)


def opd_fkl_loss(batch: FKLBatch) -> LossReport:
    """Full-information conditional forward KL, up to teacher entropy.

    Offline soft KD and on-policy FKL call this same function.  Their only
    possible difference is which states supplied the rows.
    """

    if batch.student_log_probs.shape != batch.teacher_probs.shape:
        raise ValueError("student and teacher tensors must have the same shape")
    if batch.student_log_probs.ndim < 1:
        raise ValueError("expected a final action dimension")
    teacher = batch.teacher_probs.detach()
    if bool(torch.any(teacher < 0)):
        raise ValueError("teacher probabilities must be non-negative")
    sums = teacher.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-7, rtol=1e-7):
        raise ValueError("teacher probabilities must sum to one on the final dimension")
    # A zero teacher mass contributes zero even if a deliberately
    # support-limited student uses ``-inf`` at that action.
    cross_entropy_terms = torch.where(
        teacher > 0,
        -teacher * batch.student_log_probs,
        torch.zeros_like(batch.student_log_probs),
    )
    cross_entropy = cross_entropy_terms.sum(dim=-1)
    value = masked_mean(cross_entropy, batch.mask)
    return LossReport(loss=value, metrics={"cross_entropy": value.detach()})


def rl_loss(batch: RLBatch, *, mismatch_penalty: float = 0.0) -> LossReport:
    """Unclipped importance-weighted on-policy policy-gradient surrogate."""

    if mismatch_penalty < 0:
        raise ValueError("mismatch_penalty must be non-negative")
    if not (batch.student_logp.shape == batch.behavior_logp.shape == batch.advantage.shape):
        raise ValueError("student logp, behavior logp, and advantage must have the same shape")
    behavior = batch.behavior_logp.detach()
    advantage = batch.advantage.detach()
    log_ratio = batch.student_logp - behavior
    importance_ratio = log_ratio.exp()
    per_item = -advantage * importance_ratio + mismatch_penalty * log_ratio.square()
    value = masked_mean(per_item, batch.mask)
    valid = _validated_mask(per_item, batch.mask)
    metrics = {
        "advantage": masked_mean(advantage, valid).detach(),
        "advantage_sq": masked_mean(advantage.square(), valid).detach(),
        "importance_ratio": masked_mean(importance_ratio, valid).detach(),
        "log_ratio_sq": masked_mean(log_ratio.square(), valid).detach(),
    }
    return LossReport(loss=value, metrics=metrics)


def loss(
    method: Method,
    batch: SFTBatch | RKLBatch | FKLBatch | RLBatch,
    *,
    mismatch_penalty: float = 0.0,
) -> LossReport:
    """Explicit dispatch with runtime batch-type checks."""

    expected = {
        "sft": SFTBatch,
        "opd_rkl": RKLBatch,
        "opd_fkl": FKLBatch,
        "rl": RLBatch,
    }
    if method not in expected:
        raise ValueError(f"unknown method {method!r}")
    if not isinstance(batch, expected[method]):
        raise TypeError(f"{method} expects {expected[method].__name__}, got {type(batch).__name__}")
    if method == "sft":
        return sft_loss(batch)
    if method == "opd_rkl":
        return opd_rkl_loss(batch, mismatch_penalty=mismatch_penalty)
    if method == "opd_fkl":
        return opd_fkl_loss(batch)
    return rl_loss(batch, mismatch_penalty=mismatch_penalty)


def group_advantages(rewards: Tensor, *, normalize: bool = False, eps: float = 1e-8) -> Tensor:
    """Center rewards along the final (group) dimension.

    Prime-RL's default is ``normalize=False``.  Homogeneous groups map to zero
    under either convention.
    """

    if rewards.ndim < 1 or rewards.shape[-1] < 1:
        raise ValueError("rewards must have a non-empty final group dimension")
    rewards = rewards.detach()
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not normalize:
        return centered
    scale = rewards.std(dim=-1, correction=0, keepdim=True)
    return torch.where(scale > eps, centered / scale.clamp_min(eps), torch.zeros_like(centered))


@torch.no_grad()
def sample_categorical(logits: Tensor, n: int, *, generator: torch.Generator) -> Sample:
    """Sample ``n`` actions per categorical row.

    Logits shaped ``[..., K]`` produce actions and behavior log probabilities
    shaped ``[..., n]``.  This covers independent one-step policies and each
    conditional row of a sequential policy without another sampler.
    """

    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have a final action dimension of length >= 2")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    log_probs = logits.log_softmax(dim=-1)
    action_count = logits.shape[-1]
    flat_log_probs = log_probs.reshape(-1, action_count)
    flat_actions = torch.multinomial(flat_log_probs.exp(), n, replacement=True, generator=generator)
    sampled_logp = flat_log_probs.gather(dim=-1, index=flat_actions)
    output_shape = (*logits.shape[:-1], n)
    return Sample(
        action=flat_actions.reshape(output_shape),
        behavior_logp=sampled_logp.reshape(output_shape).detach(),
    )


def categorical_kl(p: Tensor, q: Tensor) -> Tensor:
    """``KL(p || q)`` with explicit zero-support behavior."""

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if bool(torch.any(p < 0)) or bool(torch.any(q < 0)):
        raise ValueError("probabilities must be non-negative")
    p_sum = p.sum(dim=-1)
    q_sum = q.sum(dim=-1)
    if not torch.allclose(p_sum, torch.ones_like(p_sum), atol=1e-7, rtol=1e-7):
        raise ValueError("p must sum to one")
    if not torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-7, rtol=1e-7):
        raise ValueError("q must sum to one")
    log_p = torch.where(p > 0, p.log(), torch.full_like(p, -torch.inf))
    log_q = torch.where(q > 0, q.log(), torch.full_like(q, -torch.inf))
    return categorical_kl_from_log_probs(log_p, log_q)


def _validate_categorical_log_probs(name: str, log_probs: Tensor) -> None:
    if not log_probs.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if log_probs.ndim < 1 or log_probs.shape[-1] < 2:
        raise ValueError(f"{name} must have a final support dimension of length >= 2")
    invalid = torch.isnan(log_probs) | torch.isposinf(log_probs)
    if bool(torch.any(invalid)):
        raise ValueError(f"{name} may contain finite values or -inf, but not NaN or +inf")
    log_normalizer = torch.logsumexp(log_probs, dim=-1)
    if not torch.allclose(
        log_normalizer,
        torch.zeros_like(log_normalizer),
        atol=1e-7,
        rtol=1e-7,
    ):
        raise ValueError(f"{name} must be normalized along the final dimension")


def categorical_kl_from_log_probs(log_p: Tensor, log_q: Tensor) -> Tensor:
    """Stable batched ``KL(p || q)`` from normalized log probabilities.

    The result has shape ``log_p.shape[:-1]``.  Finite log masses stay finite
    even when exponentiating them would underflow to zero.  Exact ``-inf`` is
    treated as genuine zero support: a row is infinite iff ``p`` has positive
    support at an action where ``q`` is exactly zero.
    """

    if log_p.shape != log_q.shape:
        raise ValueError("log_p and log_q must have the same shape")
    _validate_categorical_log_probs("log_p", log_p)
    _validate_categorical_log_probs("log_q", log_q)

    p_has_support = ~torch.isneginf(log_p)
    q_has_support = ~torch.isneginf(log_q)
    impossible = p_has_support & ~q_has_support
    common_support = p_has_support & q_has_support

    # Form the ratio only on common support.  This avoids both ``0 * inf``
    # and ``-inf - -inf`` while retaining gradients through all finite rows.
    log_ratio = torch.where(
        common_support,
        log_p - log_q,
        torch.zeros_like(log_p),
    )
    result = (log_p.exp() * log_ratio).sum(dim=-1)
    return torch.where(impossible.any(dim=-1), torch.full_like(result, torch.inf), result)


def _validate_population_inputs(logits: Tensor, probabilities: Tensor, name: str) -> None:
    if logits.shape != probabilities.shape:
        raise ValueError(f"{name} must have the same shape as logits")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have a final action dimension of length >= 2")
    if bool(torch.any(probabilities < 0)):
        raise ValueError(f"{name} must be non-negative")
    totals = probabilities.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-7, rtol=1e-7):
        raise ValueError(f"{name} must sum to one along the final dimension")


def sft_population_values(logits: Tensor, demo_probs: Tensor) -> Tensor:
    """Per-row exact SFT cross-entropies for logits shaped ``[..., K]``."""

    _validate_population_inputs(logits, demo_probs, "demo_probs")
    return -(demo_probs.detach() * logits.log_softmax(dim=-1)).sum(dim=-1)


def sft_population_loss(logits: Tensor, demo_probs: Tensor) -> Tensor:
    """Mean exact categorical SFT/forward-KL cross-entropy."""

    return sft_population_values(logits, demo_probs).mean()


def opd_rkl_population_values(logits: Tensor, teacher_probs: Tensor) -> Tensor:
    """Per-row exact reverse KL values ``KL(p_theta || q)``."""

    _validate_population_inputs(logits, teacher_probs, "teacher_probs")
    teacher_probs = teacher_probs.detach()
    teacher_log_probs = torch.where(
        teacher_probs > 0,
        teacher_probs.log(),
        torch.full_like(teacher_probs, -torch.inf),
    )
    return categorical_kl_from_log_probs(logits.log_softmax(dim=-1), teacher_log_probs)


def opd_rkl_population_loss(logits: Tensor, teacher_probs: Tensor) -> Tensor:
    """Mean exact categorical reverse KL ``KL(p_theta || q)``."""

    return opd_rkl_population_values(logits, teacher_probs).mean()


def rl_population_values(
    logits: Tensor,
    utility: Tensor,
    *,
    reference_probs: Tensor | None = None,
    kl_beta: float = 0.0,
    entropy_bonus: float = 0.0,
) -> Tensor:
    """Per-row negative regularized utility, minimized by toy RL.

    ``loss = -E_p[utility] + kl_beta KL(p||reference) - entropy_bonus H(p)``.
    Prime-RL's rollout-policy trust penalty is not a fixed reference penalty;
    callers must opt into ``kl_beta`` deliberately.
    """

    if kl_beta < 0 or entropy_bonus < 0:
        raise ValueError("regularization coefficients must be non-negative")
    probs = logits.softmax(dim=-1)
    if utility.shape != probs.shape:
        raise ValueError("utility must have the same shape as logits")
    value = -(probs * utility.detach()).sum(dim=-1)
    if kl_beta:
        if reference_probs is None:
            raise ValueError("reference_probs is required when kl_beta > 0")
        _validate_population_inputs(logits, reference_probs, "reference_probs")
        reference_probs = reference_probs.detach()
        reference_log_probs = torch.where(
            reference_probs > 0,
            reference_probs.log(),
            torch.full_like(reference_probs, -torch.inf),
        )
        value = value + kl_beta * categorical_kl_from_log_probs(logits.log_softmax(dim=-1), reference_log_probs)
    if entropy_bonus:
        entropy = -(probs * logits.log_softmax(dim=-1)).sum(dim=-1)
        value = value - entropy_bonus * entropy
    return value


def rl_population_loss(
    logits: Tensor,
    utility: Tensor,
    *,
    reference_probs: Tensor | None = None,
    kl_beta: float = 0.0,
    entropy_bonus: float = 0.0,
) -> Tensor:
    """Mean negative regularized utility across categorical policy rows."""

    return rl_population_values(
        logits,
        utility,
        reference_probs=reference_probs,
        kl_beta=kl_beta,
        entropy_bonus=entropy_bonus,
    ).mean()


def gradient_vector(loss_value: Tensor, module: nn.Module, *, create_graph: bool = False) -> Tensor:
    """Flatten gradients in module-parameter order without mutating ``.grad``."""

    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss_value, parameters, create_graph=create_graph, allow_unused=False)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def gradient_cosine(first: Tensor, second: Tensor, *, eps: float = 1e-15) -> Tensor:
    if first.shape != second.shape:
        raise ValueError("gradient vectors must have the same shape")
    denominator = first.norm() * second.norm()
    if float(denominator.detach()) <= eps:
        return torch.zeros((), dtype=first.dtype, device=first.device)
    return torch.dot(first, second) / denominator


def _snapshot_parameters(module: nn.Module) -> tuple[list[nn.Parameter], list[Tensor]]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("module has no trainable parameters")
    return parameters, [parameter.detach().clone() for parameter in parameters]


def fixed_gradient_step_(
    module: nn.Module,
    loss_value: Tensor,
    eval_kl_from_snapshot: Callable[[], Tensor],
    *,
    step_scale: float,
    max_grad_norm: float | None = None,
) -> FixedStepReport:
    """Take one fixed-scale gradient-descent step and measure behavior KL.

    ``step_scale`` is shared across methods when comparing signal per data
    call.  ``max_grad_norm`` is a stability cap, not a normalization: vectors
    below the cap retain their raw magnitude.  The callback must evaluate
    ``KL(old policy || current policy)`` from the snapshot.  Any callback
    failure restores the original parameters.
    """

    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided")
    parameters, snapshot = _snapshot_parameters(module)
    gradients = torch.autograd.grad(loss_value, parameters, allow_unused=False)
    grad_norm = float(torch.sqrt(sum((gradient.square().sum() for gradient in gradients))))
    base_loss = float(loss_value.detach())
    if grad_norm == 0.0:
        return FixedStepReport(
            base_loss,
            0.0,
            step_scale,
            0.0,
            max_grad_norm,
            False,
            0.0,
            True,
        )

    clipped = max_grad_norm is not None and grad_norm > max_grad_norm
    clip_factor = max_grad_norm / grad_norm if clipped and max_grad_norm is not None else 1.0
    applied_scale = step_scale * clip_factor
    _set_along_direction(
        parameters,
        snapshot,
        [-gradient.detach() for gradient in gradients],
        applied_scale,
    )
    try:
        realized_kl = eval_kl_from_snapshot()
        if not isinstance(realized_kl, Tensor) or realized_kl.numel() != 1:
            raise TypeError("KL callback must return a scalar Tensor")
        if not bool(torch.isfinite(realized_kl.detach())):
            raise ValueError("KL callback returned a non-finite value")
        kl_value = float(realized_kl.detach())
        if kl_value < -1e-10:
            raise ValueError(f"KL callback returned a negative value: {kl_value}")
    except Exception:
        _set_along_direction(parameters, snapshot, gradients, 0.0)
        raise
    return FixedStepReport(
        base_loss,
        grad_norm,
        step_scale,
        applied_scale,
        max_grad_norm,
        clipped,
        max(kl_value, 0.0),
        False,
    )


@torch.no_grad()
def _set_along_direction(
    parameters: list[nn.Parameter], snapshot: list[Tensor], directions: list[Tensor], scale: float
) -> None:
    for parameter, origin, direction in zip(parameters, snapshot, directions):
        parameter.copy_(origin + scale * direction)


def kl_matched_step_(
    module: nn.Module,
    loss_value: Tensor,
    eval_kl_from_snapshot: Callable[[], Tensor],
    *,
    target_kl: float,
    initial_scale: float = 1.0,
    max_scale: float = 1e3,
    rtol: float = 1e-3,
    max_iter: int = 40,
) -> StepReport:
    """Take one gradient-descent step scaled to a fixed behavioral KL.

    ``eval_kl_from_snapshot`` must return ``KL(old policy || current policy)``
    on fixed evaluation states.  Probes always start from the same snapshot.
    If the target cannot be bracketed, parameters are restored and the report
    has ``bracketed=False`` and ``scale=0``.
    """

    if target_kl <= 0 or initial_scale <= 0 or max_scale < initial_scale:
        raise ValueError("require target_kl>0 and 0<initial_scale<=max_scale")
    parameters, snapshot = _snapshot_parameters(module)
    gradients = torch.autograd.grad(loss_value, parameters, allow_unused=False)
    directions = [-gradient.detach() for gradient in gradients]
    grad_norm = float(torch.sqrt(sum((gradient.square().sum() for gradient in gradients))))
    base_loss = float(loss_value.detach())
    if grad_norm == 0.0:
        return StepReport(base_loss, 0.0, 0.0, 0.0, target_kl, False)

    def probe(scale: float) -> float:
        _set_along_direction(parameters, snapshot, directions, scale)
        try:
            result = eval_kl_from_snapshot()
            if not isinstance(result, Tensor) or result.numel() != 1:
                raise TypeError("KL callback must return a scalar Tensor")
            if not bool(torch.isfinite(result.detach())):
                raise ValueError("KL callback returned a non-finite value")
            value = float(result.detach())
            if value < -1e-10:
                raise ValueError(f"KL callback returned a negative value: {value}")
            return max(value, 0.0)
        except Exception:
            # A failed probe must never strand the policy at a tentative step.
            _set_along_direction(parameters, snapshot, directions, 0.0)
            raise

    low, high = 0.0, initial_scale
    high_kl = probe(high)
    while high_kl < target_kl and high < max_scale:
        low = high
        high = min(2.0 * high, max_scale)
        high_kl = probe(high)
    if high_kl < target_kl:
        _set_along_direction(parameters, snapshot, directions, 0.0)
        return StepReport(base_loss, grad_norm, 0.0, 0.0, target_kl, False)

    accepted_scale, accepted_kl = high, high_kl
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_kl = probe(mid)
        accepted_scale, accepted_kl = mid, mid_kl
        if abs(mid_kl - target_kl) <= rtol * target_kl:
            break
        if mid_kl < target_kl:
            low = mid
        else:
            high = mid
    _set_along_direction(parameters, snapshot, directions, accepted_scale)
    return StepReport(base_loss, grad_norm, accepted_scale, accepted_kl, target_kl, True)


def categorical_metrics(
    student_log_probs: Tensor,
    *,
    teacher_log_probs: Tensor | None = None,
    truth_log_probs: Tensor | None = None,
    utility: Tensor | None = None,
    support_threshold: float = 1e-6,
) -> dict[str, Tensor]:
    """Common one-step distribution metrics with explicit KL orientation."""

    if student_log_probs.ndim != 1:
        raise ValueError("metrics currently expect one categorical vector")
    if not 0 < support_threshold < 1:
        raise ValueError("support_threshold must lie in (0, 1)")
    _validate_categorical_log_probs("student_log_probs", student_log_probs)
    student = student_log_probs.exp()
    entropy = -torch.special.xlogy(student, student).sum()
    metrics: dict[str, Tensor] = {
        "entropy": entropy,
        "effective_support": torch.exp(entropy),
        "support_size": (student >= support_threshold).sum(),
    }

    def add_reference(prefix: str, log_probs: Tensor) -> None:
        if log_probs.shape != student_log_probs.shape:
            raise ValueError(f"{prefix} distribution has the wrong shape")
        _validate_categorical_log_probs(f"{prefix}_log_probs", log_probs)
        reference = log_probs.exp()
        metrics[f"kl_{prefix}_student"] = categorical_kl_from_log_probs(log_probs, student_log_probs)
        metrics[f"kl_student_{prefix}"] = categorical_kl_from_log_probs(student_log_probs, log_probs)
        metrics[f"tv_{prefix}_student"] = 0.5 * (reference - student).abs().sum()
        metrics[f"{prefix}_support_recall"] = reference[student >= support_threshold].sum()
        metrics[f"{prefix}_support_precision"] = student[reference >= support_threshold].sum()

    if teacher_log_probs is not None:
        add_reference("teacher", teacher_log_probs)
    if truth_log_probs is not None:
        add_reference("truth", truth_log_probs)
    if utility is not None:
        if utility.shape != student.shape:
            raise ValueError("utility has the wrong shape")
        metrics["expected_utility"] = (student * utility).sum()
    return {key: value.detach() for key, value in metrics.items()}


def normalized_grid_log_mass(log_density: Tensor, *, dx: float | Tensor) -> Tensor:
    """Convert log density values to normalized quadrature-bin log masses.

    ``dx`` may be a scalar for an equally spaced grid or positive tensor of
    quadrature weights broadcastable to ``log_density``.
    """

    if log_density.ndim < 1:
        raise ValueError("log_density must be non-empty")
    dx_tensor = torch.as_tensor(dx, dtype=log_density.dtype, device=log_density.device)
    if bool(torch.any(dx_tensor <= 0)):
        raise ValueError("dx must contain only positive quadrature weights")
    try:
        log_unnormalized_mass = log_density + dx_tensor.log()
    except RuntimeError as error:
        raise ValueError("dx must broadcast to log_density") from error
    return log_unnormalized_mass - torch.logsumexp(log_unnormalized_mass, dim=-1, keepdim=True)
