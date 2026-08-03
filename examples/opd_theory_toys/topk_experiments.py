"""First-principles experiments for Prime's teacher-top-k OPD estimator.

Let ``H`` be the teacher top-k, ``T`` its complement, ``p`` the current
student, and ``b`` the (detached) rollout distribution.  Prime's augmented
support loss for a sampled token ``y`` is

    L_union(y) = sum_{i in H} f_i + 1[y in T] f_y,
    f_i = p_i log(p_i / q_i).

Consequently, exact enumeration over ``y ~ b`` gives coefficients ``1`` on
``H`` and ``b_i`` on ``T``.  For fresh on-policy samples, ``b=p``, so omitted
terms scale as ``p_i**2`` instead of ``p_i``.  This module turns that identity
into a predictive gradient formula, checks it against Prime-RL's real
``opd_loss_fn``, and maps the alignment boundary.

The importance-weighted control is

    L_IW(y) = sum_{i in H} f_i + 1[y in T] f_y / stop_gradient(b_y).

It is exactly unbiased when every omitted token has positive proposal mass.
Its variance can be large, which is reported alongside its mean gradient.

A lower-information control collapses the omitted support into one residual
bucket with masses ``P_T=sum_T p_i`` and ``Q_T=sum_T q_i``:

    L_bucket = sum_{i in H} f_i + P_T log(P_T / Q_T).

This is a deterministic coarse-grained KL.  It is nonnegative and stationary
at ``p=q``, but cannot distinguish reallocations that preserve total tail mass.
No SFT, RL, or OPD loss is reimplemented here; implementation-level checks
compose the loss function from the current Prime-RL checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Literal, Sequence

import torch
from torch import Tensor

from prime_rl.trainer.rl.loss import opd_loss_fn

try:  # Support both package imports and ``python topk_experiments.py``.
    from .prime_loss_checks import DTYPE, _analytic_reverse_kl, _logits_for, _one_token_inputs, _probabilities
except ImportError:
    from prime_loss_checks import DTYPE, _analytic_reverse_kl, _logits_for, _one_token_inputs, _probabilities


Estimator = Literal["union", "importance_weighted_tail", "residual_bucket"]


@dataclass(frozen=True)
class PrimeEnumeration:
    """Exact expectation of one Prime estimator over all rollout tokens."""

    estimator: Estimator
    expected_loss: Tensor
    gradient: Tensor
    teacher_topk_ids: tuple[int, ...]


@dataclass(frozen=True)
class FiniteDifferenceCheck:
    """Autograd and central-finite-difference gradients of the real loss."""

    estimator: Estimator
    autograd_gradient: Tensor
    finite_difference_gradient: Tensor

    @property
    def max_abs_error(self) -> float:
        return float((self.autograd_gradient - self.finite_difference_gradient).abs().max())


@dataclass(frozen=True)
class EstimatorMoments:
    """Exact one-sample gradient mean and total coordinate variance."""

    estimator: Estimator
    mean_gradient: Tensor
    variance_trace: float

    @property
    def rms_noise(self) -> float:
        return sqrt(self.variance_trace)


@dataclass(frozen=True)
class TopKComparison:
    """Full-RKL comparison for one arbitrary categorical student/teacher pair."""

    scenario: str
    k: int
    vocabulary_size: int
    teacher_topk_ids: tuple[int, ...]
    omitted_ids: tuple[int, ...]
    omitted_student_mass: float
    omitted_teacher_mass: float
    omitted_collision_mass: float
    omitted_effective_modes: float
    omitted_mean_log_p_over_q: float
    full_gradient_norm: float
    union_gradient_norm: float
    relative_bias: float
    cosine_similarity: float
    directional_gain: float
    relation: str
    residual_bucket_loss: float
    residual_bucket_gradient_norm: float
    residual_bucket_relative_bias: float
    residual_bucket_cosine_similarity: float
    residual_bucket_directional_gain: float
    residual_bucket_relation: str
    union_relative_rms_noise: float
    importance_weighted_relative_rms_noise: float
    prime_formula_max_error: float | None
    importance_weighted_max_error: float | None
    residual_bucket_max_error: float | None


@dataclass(frozen=True)
class SymmetricBoundaryRow:
    """Closed-form phase-boundary row for a uniform covered head and tail."""

    omitted_student_mass: float
    tail_log_p_over_q: float
    head_log_p_over_q: float
    k: int
    omitted_modes: int
    full_tail_margin: float
    union_tail_margin: float
    full_gradient_norm: float
    union_gradient_norm: float
    cosine_similarity: float
    directional_gain: float
    relative_bias: float
    relation: str
    residual_bucket_directional_gain: float
    residual_bucket_relative_bias: float
    residual_bucket_relation: str
    union_relative_rms_noise: float
    importance_weighted_relative_rms_noise: float
    prime_formula_max_error: float | None
    importance_weighted_max_error: float | None
    residual_bucket_max_error: float | None


def _as_probabilities(values: Sequence[float] | Tensor) -> Tensor:
    if isinstance(values, Tensor):
        values = values.detach().cpu().tolist()
    return _probabilities(values)


def _checked_pair(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
) -> tuple[Tensor, Tensor]:
    student_p = _as_probabilities(student)
    teacher_p = _as_probabilities(teacher)
    if student_p.shape != teacher_p.shape:
        raise ValueError(
            f"Student and teacher must have the same vocabulary; got {student_p.shape} and {teacher_p.shape}."
        )
    return student_p, teacher_p


def _checked_proposal(proposal: Sequence[float] | Tensor | None, student_p: Tensor) -> Tensor:
    if proposal is None:
        return student_p.clone()
    proposal_p = _as_probabilities(proposal)
    if proposal_p.shape != student_p.shape:
        raise ValueError(f"Proposal shape {proposal_p.shape} does not match student shape {student_p.shape}.")
    return proposal_p


def _teacher_topk(teacher_p: Tensor, k: int) -> Tensor:
    if not 1 <= k <= teacher_p.numel():
        raise ValueError(f"k must lie in [1, {teacher_p.numel()}], got {k}.")
    return teacher_p.topk(k).indices


def full_reverse_kl_gradient(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
) -> Tensor:
    """Analytic logit gradient of ``KL(student || teacher)``."""

    student_p, teacher_p = _checked_pair(student, teacher)
    return _analytic_reverse_kl(student_p, teacher_p)[1]


def expected_union_gradient_prediction(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    *,
    k: int,
    proposal: Sequence[float] | Tensor | None = None,
) -> Tensor:
    """Predict the exact expected Prime top-k-union logit gradient.

    For ``h_i = 1 + log(p_i/q_i)`` and coefficients

    ``a_i = 1`` on teacher top-k, ``a_i = b_i`` otherwise,

    the result is ``g_j = p_j [a_j h_j - sum_i a_i p_i h_i]``.
    ``proposal=None`` means fresh on-policy rollouts, ``b=p``.
    """

    student_p, teacher_p = _checked_pair(student, teacher)
    proposal_p = _checked_proposal(proposal, student_p)
    topk_ids = _teacher_topk(teacher_p, k)
    coefficient = proposal_p.clone()
    coefficient[topk_ids] = 1.0
    h = 1.0 + student_p.log() - teacher_p.log()
    centering = (coefficient * student_p * h).sum()
    return student_p * (coefficient * h - centering)


def residual_bucket_loss_and_gradient(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    *,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Return the coarse-grained KL and its logit gradient.

    Teacher-top-k outcomes remain individual categories and all omitted
    outcomes become one residual category.  The resulting distribution has
    ``k+1`` categories (or ``k`` when ``k`` is the full vocabulary), so this
    objective is a genuine KL and is deterministic.  Only the aggregate tail
    masses are needed; conditional allocation inside the tail is discarded.
    """

    student_p, teacher_p = _checked_pair(student, teacher)
    topk_ids = _teacher_topk(teacher_p, k)
    logits = _logits_for(student_p)
    student_logp = logits.log_softmax(dim=-1)
    teacher_logp = teacher_p.log()
    student_topk_p = student_logp[topk_ids].exp()
    topk_loss = (student_topk_p * (student_logp[topk_ids] - teacher_logp[topk_ids])).sum()
    if k == student_p.numel():
        loss = topk_loss
    else:
        student_tail_mass = 1.0 - student_topk_p.sum()
        teacher_tail_mass = 1.0 - teacher_p[topk_ids].sum()
        tail_bucket_loss = student_tail_mass * (student_tail_mass.log() - teacher_tail_mass.log())
        loss = topk_loss + tail_bucket_loss
    gradient = torch.autograd.grad(loss, logits)[0]
    return loss.detach(), gradient.detach()


def _prime_expected_loss_from_logits(
    logits: Tensor,
    teacher_p: Tensor,
    proposal_p: Tensor,
    teacher_topk_ids: Tensor,
    *,
    estimator: Estimator,
) -> Tensor:
    """Enumerate rollout tokens while delegating every OPD term to Prime."""

    student_logp = logits.log_softmax(dim=-1)
    teacher_logp = teacher_p.log()
    if estimator == "residual_bucket":
        sampled_token = int(teacher_topk_ids[0])
        output = opd_loss_fn(
            _one_token_inputs(
                student_logp[sampled_token],
                inference_logprob=proposal_p[sampled_token].log(),
                teacher_logprob=teacher_logp[sampled_token],
                teacher_topk_logprobs=teacher_logp[teacher_topk_ids],
                student_topk_logprobs=student_logp[teacher_topk_ids],
                sampled_token_in_teacher_topk=True,
            )
        )
        topk_base = output.metrics["topk_reverse_kl"].sum()
        if teacher_topk_ids.numel() == student_logp.numel():
            return topk_base
        student_tail_mass = 1.0 - student_logp[teacher_topk_ids].exp().sum()
        teacher_tail_mass = 1.0 - teacher_p[teacher_topk_ids].sum()
        return topk_base + student_tail_mass * (student_tail_mass.log() - teacher_tail_mass.log())

    expected_loss = torch.zeros((), dtype=DTYPE)

    for sampled_token in range(student_logp.numel()):
        sampled_in_topk = bool(torch.any(teacher_topk_ids == sampled_token))
        output = opd_loss_fn(
            _one_token_inputs(
                student_logp[sampled_token],
                inference_logprob=proposal_p[sampled_token].log(),
                teacher_logprob=teacher_logp[sampled_token],
                teacher_topk_logprobs=teacher_logp[teacher_topk_ids],
                student_topk_logprobs=student_logp[teacher_topk_ids],
                sampled_token_in_teacher_topk=sampled_in_topk,
            )
        )
        sample_loss = output.loss
        if estimator == "importance_weighted_tail":
            topk_base = output.metrics["topk_reverse_kl"].sum()
            sampled_tail_increment = sample_loss - topk_base
            sample_loss = topk_base + sampled_tail_increment / proposal_p[sampled_token]
        elif estimator != "union":
            raise ValueError(f"Unknown estimator: {estimator}.")
        expected_loss = expected_loss + proposal_p[sampled_token] * sample_loss
    return expected_loss


def enumerate_prime_gradient(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    *,
    k: int,
    estimator: Estimator = "union",
    proposal: Sequence[float] | Tensor | None = None,
) -> PrimeEnumeration:
    """Exactly enumerate the expected gradient of Prime's current loss.

    The proposal is held fixed, as rollout data are in training.  For the
    importance-weighted estimator, the sampled tail increment produced by
    Prime is divided by its detached proposal probability.  This requires
    full support, which is already enforced by the categorical validator.
    """

    student_p, teacher_p = _checked_pair(student, teacher)
    proposal_p = _checked_proposal(proposal, student_p)
    teacher_topk_ids = _teacher_topk(teacher_p, k)
    logits = _logits_for(student_p)
    loss = _prime_expected_loss_from_logits(
        logits,
        teacher_p,
        proposal_p,
        teacher_topk_ids,
        estimator=estimator,
    )
    gradient = torch.autograd.grad(loss, logits)[0]
    return PrimeEnumeration(
        estimator=estimator,
        expected_loss=loss.detach(),
        gradient=gradient.detach(),
        teacher_topk_ids=tuple(int(index) for index in teacher_topk_ids.tolist()),
    )


def finite_difference_prime_gradient(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    *,
    k: int,
    estimator: Estimator = "union",
    proposal: Sequence[float] | Tensor | None = None,
    epsilon: float = 1e-6,
) -> FiniteDifferenceCheck:
    """Central-difference check of the exact-enumeration Prime surrogate."""

    if not isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}.")
    student_p, teacher_p = _checked_pair(student, teacher)
    proposal_p = _checked_proposal(proposal, student_p)
    teacher_topk_ids = _teacher_topk(teacher_p, k)
    reference_logits = student_p.log()
    autograd_logits = reference_logits.clone().detach().requires_grad_(True)
    loss = _prime_expected_loss_from_logits(
        autograd_logits,
        teacher_p,
        proposal_p,
        teacher_topk_ids,
        estimator=estimator,
    )
    autograd_gradient = torch.autograd.grad(loss, autograd_logits)[0].detach()

    finite_difference = torch.empty_like(reference_logits)
    for coordinate in range(reference_logits.numel()):
        step = torch.zeros_like(reference_logits)
        step[coordinate] = epsilon
        loss_plus = _prime_expected_loss_from_logits(
            reference_logits + step,
            teacher_p,
            proposal_p,
            teacher_topk_ids,
            estimator=estimator,
        )
        loss_minus = _prime_expected_loss_from_logits(
            reference_logits - step,
            teacher_p,
            proposal_p,
            teacher_topk_ids,
            estimator=estimator,
        )
        finite_difference[coordinate] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return FiniteDifferenceCheck(
        estimator=estimator,
        autograd_gradient=autograd_gradient,
        finite_difference_gradient=finite_difference.detach(),
    )


def estimator_gradient_moments(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    *,
    k: int,
    estimator: Estimator = "union",
    proposal: Sequence[float] | Tensor | None = None,
) -> EstimatorMoments:
    """Return exact one-rollout gradient mean and variance.

    This is the score variance of the sparse loss with the proposal held
    fixed, not the variance of sampling the proposal itself.  The IW mean is
    the full-RKL gradient, but a token with proposal mass ``b_i`` contributes
    a term of size ``1/b_i`` when sampled.
    """

    student_p, teacher_p = _checked_pair(student, teacher)
    proposal_p = _checked_proposal(proposal, student_p)
    if estimator == "residual_bucket":
        _, gradient = residual_bucket_loss_and_gradient(student_p, teacher_p, k=k)
        return EstimatorMoments(estimator=estimator, mean_gradient=gradient, variance_trace=0.0)

    topk_ids = _teacher_topk(teacher_p, k)
    in_topk = torch.zeros(student_p.numel(), dtype=torch.bool)
    in_topk[topk_ids] = True

    h = 1.0 + student_p.log() - teacher_p.log()
    identity = torch.eye(student_p.numel(), dtype=DTYPE)
    term_gradients = student_p[:, None] * h[:, None] * (identity - student_p[None, :])
    topk_base_gradient = term_gradients[in_topk].sum(dim=0)
    sample_gradients = topk_base_gradient.repeat(student_p.numel(), 1)
    tail_ids = (~in_topk).nonzero(as_tuple=False).flatten()
    if estimator == "union":
        sample_gradients[tail_ids] += term_gradients[tail_ids]
    elif estimator == "importance_weighted_tail":
        sample_gradients[tail_ids] += term_gradients[tail_ids] / proposal_p[tail_ids, None]
    else:
        raise ValueError(f"Unknown estimator: {estimator}.")

    mean = (proposal_p[:, None] * sample_gradients).sum(dim=0)
    centered_norm_sq = (sample_gradients - mean).square().sum(dim=1)
    variance_trace = float((proposal_p * centered_norm_sq).sum())
    return EstimatorMoments(estimator=estimator, mean_gradient=mean, variance_trace=variance_trace)


def _gradient_relation(full_gradient: Tensor, candidate_gradient: Tensor) -> tuple[float, float, str]:
    full_norm = torch.linalg.vector_norm(full_gradient)
    candidate_norm = torch.linalg.vector_norm(candidate_gradient)
    norm_product = full_norm * candidate_norm
    if float(full_norm) <= 1e-14:
        return float("nan"), float("nan"), "full-stationary"
    if float(candidate_norm) <= 1e-14:
        return float("nan"), 0.0, "stalled"
    dot = torch.dot(full_gradient, candidate_gradient)
    cosine = float(dot / norm_product)
    gain = float(dot / full_norm.square())
    if cosine > 1e-10:
        relation = "aligns"
    elif cosine < -1e-10:
        relation = "opposes"
    else:
        relation = "orthogonal"
    return cosine, gain, relation


def compare_topk_gradients(
    student: Sequence[float] | Tensor,
    teacher: Sequence[float] | Tensor,
    *,
    k: int,
    scenario: str = "categorical",
    proposal: Sequence[float] | Tensor | None = None,
    verify_prime: bool = False,
) -> TopKComparison:
    """Compare predicted union and IW gradients with full reverse KL."""

    student_p, teacher_p = _checked_pair(student, teacher)
    proposal_p = _checked_proposal(proposal, student_p)
    topk_ids = _teacher_topk(teacher_p, k)
    in_topk = torch.zeros(student_p.numel(), dtype=torch.bool)
    in_topk[topk_ids] = True
    omitted_ids_tensor = (~in_topk).nonzero(as_tuple=False).flatten()

    full_gradient = _analytic_reverse_kl(student_p, teacher_p)[1]
    union_gradient = expected_union_gradient_prediction(
        student_p,
        teacher_p,
        k=k,
        proposal=proposal_p,
    )
    union_moments = estimator_gradient_moments(
        student_p,
        teacher_p,
        k=k,
        estimator="union",
        proposal=proposal_p,
    )
    iw_moments = estimator_gradient_moments(
        student_p,
        teacher_p,
        k=k,
        estimator="importance_weighted_tail",
        proposal=proposal_p,
    )
    bucket_loss, bucket_gradient = residual_bucket_loss_and_gradient(student_p, teacher_p, k=k)
    cosine, directional_gain, relation = _gradient_relation(full_gradient, union_gradient)
    bucket_cosine, bucket_directional_gain, bucket_relation = _gradient_relation(full_gradient, bucket_gradient)
    full_norm = torch.linalg.vector_norm(full_gradient)
    union_norm = torch.linalg.vector_norm(union_gradient)
    bucket_norm = torch.linalg.vector_norm(bucket_gradient)
    bias_norm = torch.linalg.vector_norm(union_gradient - full_gradient)
    bucket_bias_norm = torch.linalg.vector_norm(bucket_gradient - full_gradient)

    omitted_student_mass = student_p[omitted_ids_tensor].sum()
    omitted_teacher_mass = teacher_p[omitted_ids_tensor].sum()
    collision_mass = student_p[omitted_ids_tensor].square().sum()
    if float(omitted_student_mass) > 0.0:
        mean_log_ratio = (
            student_p[omitted_ids_tensor] * (student_p[omitted_ids_tensor].log() - teacher_p[omitted_ids_tensor].log())
        ).sum() / omitted_student_mass
        effective_modes = omitted_student_mass.square() / collision_mass
    else:
        mean_log_ratio = torch.tensor(float("nan"), dtype=DTYPE)
        effective_modes = torch.tensor(0.0, dtype=DTYPE)

    prime_formula_error = None
    importance_weighted_error = None
    residual_bucket_error = None
    if verify_prime:
        prime_union = enumerate_prime_gradient(
            student_p,
            teacher_p,
            k=k,
            estimator="union",
            proposal=proposal_p,
        )
        prime_iw = enumerate_prime_gradient(
            student_p,
            teacher_p,
            k=k,
            estimator="importance_weighted_tail",
            proposal=proposal_p,
        )
        prime_bucket = enumerate_prime_gradient(
            student_p,
            teacher_p,
            k=k,
            estimator="residual_bucket",
            proposal=proposal_p,
        )
        prime_formula_error = float((prime_union.gradient - union_gradient).abs().max())
        importance_weighted_error = float((prime_iw.gradient - full_gradient).abs().max())
        residual_bucket_error = float((prime_bucket.gradient - bucket_gradient).abs().max())

    if float(full_norm) > 0.0:
        relative_bias = float(bias_norm / full_norm)
        bucket_relative_bias = float(bucket_bias_norm / full_norm)
        union_relative_noise = union_moments.rms_noise / float(full_norm)
        iw_relative_noise = iw_moments.rms_noise / float(full_norm)
    else:
        relative_bias = float("nan")
        bucket_relative_bias = float("nan")
        union_relative_noise = float("nan")
        iw_relative_noise = float("nan")

    return TopKComparison(
        scenario=scenario,
        k=k,
        vocabulary_size=student_p.numel(),
        teacher_topk_ids=tuple(int(index) for index in topk_ids.tolist()),
        omitted_ids=tuple(int(index) for index in omitted_ids_tensor.tolist()),
        omitted_student_mass=float(omitted_student_mass),
        omitted_teacher_mass=float(omitted_teacher_mass),
        omitted_collision_mass=float(collision_mass),
        omitted_effective_modes=float(effective_modes),
        omitted_mean_log_p_over_q=float(mean_log_ratio),
        full_gradient_norm=float(full_norm),
        union_gradient_norm=float(union_norm),
        relative_bias=relative_bias,
        cosine_similarity=cosine,
        directional_gain=directional_gain,
        relation=relation,
        residual_bucket_loss=float(bucket_loss),
        residual_bucket_gradient_norm=float(bucket_norm),
        residual_bucket_relative_bias=bucket_relative_bias,
        residual_bucket_cosine_similarity=bucket_cosine,
        residual_bucket_directional_gain=bucket_directional_gain,
        residual_bucket_relation=bucket_relation,
        union_relative_rms_noise=union_relative_noise,
        importance_weighted_relative_rms_noise=iw_relative_noise,
        prime_formula_max_error=prime_formula_error,
        importance_weighted_max_error=importance_weighted_error,
        residual_bucket_max_error=residual_bucket_error,
    )


def symmetric_head_tail_distributions(
    omitted_student_mass: float,
    tail_log_p_over_q: float,
    *,
    k: int,
    omitted_modes: int,
) -> tuple[Tensor, Tensor]:
    """Build the minimal distribution with independently controlled geometry.

    The first ``k`` tokens form a uniform covered head.  The remaining
    ``omitted_modes`` tokens share student mass ``M`` uniformly and have a
    common ``log(p_i/q_i)=r``.  Normalization fixes the covered-head log ratio.
    A strict ranking check ensures the intended head really is teacher top-k.
    """

    if not 0.0 < omitted_student_mass < 1.0:
        raise ValueError(f"omitted_student_mass must lie in (0, 1), got {omitted_student_mass}.")
    if not isfinite(tail_log_p_over_q):
        raise ValueError(f"tail_log_p_over_q must be finite, got {tail_log_p_over_q}.")
    if k < 1 or omitted_modes < 1:
        raise ValueError(f"k and omitted_modes must be positive, got k={k}, omitted_modes={omitted_modes}.")

    teacher_tail_mass = omitted_student_mass * exp(-tail_log_p_over_q)
    if not 0.0 < teacher_tail_mass < 1.0:
        raise ValueError(
            "The requested tail log-ratio implies invalid teacher tail mass "
            f"{teacher_tail_mass:.6g}; require 0 < M exp(-r) < 1."
        )
    student_head_probability = (1.0 - omitted_student_mass) / k
    teacher_head_probability = (1.0 - teacher_tail_mass) / k
    student_tail_probability = omitted_student_mass / omitted_modes
    teacher_tail_probability = teacher_tail_mass / omitted_modes
    if teacher_head_probability <= teacher_tail_probability:
        raise ValueError(
            "The intended covered head is not teacher top-k: each teacher head token has mass "
            f"{teacher_head_probability:.6g}, versus tail mass {teacher_tail_probability:.6g}."
        )

    student = [student_head_probability] * k + [student_tail_probability] * omitted_modes
    teacher = [teacher_head_probability] * k + [teacher_tail_probability] * omitted_modes
    return _probabilities(student), _probabilities(teacher)


def symmetric_boundary_row(
    omitted_student_mass: float,
    tail_log_p_over_q: float,
    *,
    k: int,
    omitted_modes: int,
    verify_prime: bool = False,
) -> SymmetricBoundaryRow:
    """Evaluate the exact two-block alignment boundary.

    Let ``M`` be omitted student mass, ``m`` the number of equally likely
    omitted modes, ``r_T=log(p_T/q_T)``, and ``r_H`` the covered-head ratio.
    The aggregate tail components of the two logit gradients are proportional
    to

    ``full_margin  = r_T - r_H``

    ``union_margin = (M/m) (1+r_T) - (1+r_H)``.

    Their product determines align versus oppose.  At fixed mass and ratio,
    increasing tail fragmentation ``m`` pushes the union margin downward.
    The number of covered tokens ``k`` matters only through which outcomes the
    teacher ranking covers; once the same aggregate head is covered, it drops
    out of this symmetric boundary.
    """

    student_p, teacher_p = symmetric_head_tail_distributions(
        omitted_student_mass,
        tail_log_p_over_q,
        k=k,
        omitted_modes=omitted_modes,
    )
    teacher_tail_mass = omitted_student_mass * exp(-tail_log_p_over_q)
    head_log_ratio = log((1.0 - omitted_student_mass) / (1.0 - teacher_tail_mass))
    full_tail_margin = tail_log_p_over_q - head_log_ratio
    union_tail_margin = omitted_student_mass / omitted_modes * (1.0 + tail_log_p_over_q) - (1.0 + head_log_ratio)
    comparison = compare_topk_gradients(
        student_p,
        teacher_p,
        k=k,
        scenario=f"uniform-head/{omitted_modes}-tail-modes",
        verify_prime=verify_prime,
    )
    return SymmetricBoundaryRow(
        omitted_student_mass=omitted_student_mass,
        tail_log_p_over_q=tail_log_p_over_q,
        head_log_p_over_q=head_log_ratio,
        k=k,
        omitted_modes=omitted_modes,
        full_tail_margin=full_tail_margin,
        union_tail_margin=union_tail_margin,
        full_gradient_norm=comparison.full_gradient_norm,
        union_gradient_norm=comparison.union_gradient_norm,
        cosine_similarity=comparison.cosine_similarity,
        directional_gain=comparison.directional_gain,
        relative_bias=comparison.relative_bias,
        relation=comparison.relation,
        residual_bucket_directional_gain=comparison.residual_bucket_directional_gain,
        residual_bucket_relative_bias=comparison.residual_bucket_relative_bias,
        residual_bucket_relation=comparison.residual_bucket_relation,
        union_relative_rms_noise=comparison.union_relative_rms_noise,
        importance_weighted_relative_rms_noise=comparison.importance_weighted_relative_rms_noise,
        prime_formula_max_error=comparison.prime_formula_max_error,
        importance_weighted_max_error=comparison.importance_weighted_max_error,
        residual_bucket_max_error=comparison.residual_bucket_max_error,
    )


def symmetric_boundary_sweep(
    omitted_student_masses: Sequence[float] = (0.01, 0.1, 0.3),
    tail_log_p_over_q_values: Sequence[float] = (0.5, 2.0, 5.0, 10.0, 20.0),
    *,
    k_values: Sequence[int] = (1, 2, 4),
    omitted_mode_counts: Sequence[int] = (1, 4, 16),
) -> list[SymmetricBoundaryRow]:
    """Return pivot-ready rows over mass, mismatch, k, and fragmentation."""

    return [
        symmetric_boundary_row(mass, log_ratio, k=k, omitted_modes=modes)
        for mass in omitted_student_masses
        for log_ratio in tail_log_p_over_q_values
        for k in k_values
        for modes in omitted_mode_counts
    ]


def ranked_mode_k_sweep(
    student: Sequence[float] | Tensor = (0.35, 0.25, 0.10, 0.10, 0.10, 0.10),
    teacher: Sequence[float] | Tensor = (0.45, 0.25, 0.15, 0.08, 0.05, 0.02),
    *,
    verify_prime: bool = False,
) -> list[TopKComparison]:
    """Sweep k on one fixed ranked multi-mode distribution.

    Each category is a mode.  The default teacher ranks modes from common to
    rare, while the student overweights the two rarest modes.  Holding the
    pair fixed isolates the effect of changing ``k`` rather than changing the
    omitted mass and mismatch at the same time.
    """

    student_p, teacher_p = _checked_pair(student, teacher)
    return [
        compare_topk_gradients(
            student_p,
            teacher_p,
            k=k,
            scenario="ranked-six-mode",
            verify_prime=verify_prime,
        )
        for k in range(1, student_p.numel() + 1)
    ]


def teacher_match_k_sweep(
    teacher: Sequence[float] | Tensor = (0.45, 0.25, 0.15, 0.08, 0.05, 0.02),
    *,
    verify_prime: bool = False,
) -> list[TopKComparison]:
    """Measure spurious sparse-gradient drift at the exact teacher match.

    Full reverse KL is stationary when ``student=teacher``.  A partial sum of
    full-normalized categorical terms is not: unless ``k`` covers the full
    vocabulary, cancellation between token gradients is broken.  The unioned
    rollout token only partially restores it because omitted token ``i`` is
    included with probability ``p_i``.
    """

    teacher_p = _as_probabilities(teacher)
    return [
        compare_topk_gradients(
            teacher_p,
            teacher_p,
            k=k,
            scenario="teacher-match-null",
            verify_prime=verify_prime,
        )
        for k in range(1, teacher_p.numel() + 1)
    ]


def residual_bucket_blind_spot(*, verify_prime: bool = False) -> TopKComparison:
    """Return a case where only within-tail allocation is wrong.

    Student and teacher agree on both top-2 probabilities and total residual
    mass, but swap the conditional masses of the two omitted modes.  The
    residual-bucket KL is exactly zero and therefore cannot distinguish them;
    full RKL and the importance-weighted estimator can.
    """

    return compare_topk_gradients(
        student=(0.60, 0.25, 0.05, 0.10),
        teacher=(0.60, 0.25, 0.10, 0.05),
        k=2,
        scenario="within-tail-swap",
        verify_prime=verify_prime,
    )


def compact_comparison_records(rows: Sequence[TopKComparison]) -> list[dict[str, float | int | str]]:
    """Select notebook-friendly columns without requiring pandas."""

    return [
        {
            "scenario": row.scenario,
            "k": row.k,
            "vocab": row.vocabulary_size,
            "omitted_p_mass": row.omitted_student_mass,
            "omitted_q_mass": row.omitted_teacher_mass,
            "tail_log_p_over_q": row.omitted_mean_log_p_over_q,
            "tail_log_q_over_p": -row.omitted_mean_log_p_over_q,
            "tail_collision_mass": row.omitted_collision_mass,
            "effective_tail_modes": row.omitted_effective_modes,
            "full_gradient_norm": row.full_gradient_norm,
            "union_gradient_norm": row.union_gradient_norm,
            "cosine": row.cosine_similarity,
            "directional_gain": row.directional_gain,
            "relative_bias": row.relative_bias,
            "bucket_loss": row.residual_bucket_loss,
            "bucket_gradient_norm": row.residual_bucket_gradient_norm,
            "bucket_cosine": row.residual_bucket_cosine_similarity,
            "bucket_directional_gain": row.residual_bucket_directional_gain,
            "bucket_relative_bias": row.residual_bucket_relative_bias,
            "bucket_relation": row.residual_bucket_relation,
            "union_rel_noise": row.union_relative_rms_noise,
            "iw_rel_noise": row.importance_weighted_relative_rms_noise,
            "relation": row.relation,
        }
        for row in rows
    ]


def compact_boundary_records(rows: Sequence[SymmetricBoundaryRow]) -> list[dict[str, float | int | str]]:
    """Select phase-map columns for a dataframe or plotting library."""

    return [
        {
            "omitted_p_mass": row.omitted_student_mass,
            "tail_log_p_over_q": row.tail_log_p_over_q,
            "tail_log_q_over_p": -row.tail_log_p_over_q,
            "k": row.k,
            "tail_modes": row.omitted_modes,
            "full_margin": row.full_tail_margin,
            "union_margin": row.union_tail_margin,
            "full_gradient_norm": row.full_gradient_norm,
            "union_gradient_norm": row.union_gradient_norm,
            "cosine": row.cosine_similarity,
            "directional_gain": row.directional_gain,
            "relative_bias": row.relative_bias,
            "bucket_directional_gain": row.residual_bucket_directional_gain,
            "bucket_relative_bias": row.residual_bucket_relative_bias,
            "bucket_relation": row.residual_bucket_relation,
            "union_rel_noise": row.union_relative_rms_noise,
            "iw_rel_noise": row.importance_weighted_relative_rms_noise,
            "relation": row.relation,
        }
        for row in rows
    ]


def _format_float(value: float, width: int = 10) -> str:
    if value != value:
        return f"{'nan':>{width}}"
    return f"{value:{width}.3g}"


def format_boundary_table(rows: Sequence[SymmetricBoundaryRow]) -> str:
    """Format a compact phase-boundary table for terminal/notebook display."""

    header = " M_tail   log(p/q)   k   modes  full margin  union margin  union gain  bucket gain  relation"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.omitted_student_mass:7.3g}  {row.tail_log_p_over_q:9.3g}  {row.k:2d}  "
            f"{row.omitted_modes:6d}  {_format_float(row.full_tail_margin, 11)}  "
            f"{_format_float(row.union_tail_margin, 12)}  {_format_float(row.directional_gain, 10)}  "
            f"{_format_float(row.residual_bucket_directional_gain, 11)}  {row.relation}"
        )
    return "\n".join(lines)


def format_comparison_table(rows: Sequence[TopKComparison]) -> str:
    """Format arbitrary k-sweep comparisons."""

    header = " k   omitted p   omitted q   eff modes   cosine      gain   rel bias  relation"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.k:2d}  {row.omitted_student_mass:10.3g}  {row.omitted_teacher_mass:10.3g}  "
            f"{row.omitted_effective_modes:9.3g}  {_format_float(row.cosine_similarity, 7)}  "
            f"{_format_float(row.directional_gain, 8)}  {_format_float(row.relative_bias, 9)}  "
            f"{row.relation}"
        )
    return "\n".join(lines)


def format_teacher_match_table(rows: Sequence[TopKComparison]) -> str:
    """Format the ``p=q`` fixed-point diagnostic."""

    header = " k   omitted p   ||g_full||  ||g_union||  relation"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.k:2d}  {row.omitted_student_mass:10.3g}  {_format_float(row.full_gradient_norm, 10)}  "
            f"{_format_float(row.union_gradient_norm, 11)}  {row.relation}"
        )
    return "\n".join(lines)


def format_control_comparison_table(rows: Sequence[TopKComparison]) -> str:
    """Contrast biased union, coarse bucket, and unbiased IW controls."""

    header = " scenario              k  union gain  union noise  bucket gain  bucket relation   IW noise"
    lines = [header]
    for row in rows:
        lines.append(
            f" {row.scenario:<21} {row.k:2d}  {_format_float(row.directional_gain, 10)}  "
            f"{_format_float(row.union_relative_rms_noise, 11)}  "
            f"{_format_float(row.residual_bucket_directional_gain, 11)}  "
            f"{row.residual_bucket_relation:<15}  "
            f"{_format_float(row.importance_weighted_relative_rms_noise, 8)}"
        )
    return "\n".join(lines)


def representative_boundary_rows(*, verify_prime: bool = True) -> list[SymmetricBoundaryRow]:
    """Small table spanning both sides of the predicted phase boundary."""

    settings = (
        (0.10, 0.0, 2, 1),
        (0.01, 2.0, 2, 1),
        (0.10, 2.0, 2, 1),
        (0.10, 10.0, 2, 1),
        (0.10, 10.0, 2, 4),
        (0.30, 2.0, 2, 1),
        (0.30, 2.0, 2, 4),
    )
    return [
        symmetric_boundary_row(mass, ratio, k=k, omitted_modes=modes, verify_prime=verify_prime)
        for mass, ratio, k, modes in settings
    ]


def run_default_experiments() -> dict[str, Sequence[SymmetricBoundaryRow] | Sequence[TopKComparison]]:
    """Run deterministic CPU experiments and return inspectable table rows."""

    return {
        "representative_boundary": representative_boundary_rows(verify_prime=True),
        "ranked_mode_k": ranked_mode_k_sweep(verify_prime=True),
        "teacher_match_k": teacher_match_k_sweep(verify_prime=True),
        "bucket_blind_spot": [residual_bucket_blind_spot(verify_prime=True)],
        "full_boundary_map": symmetric_boundary_sweep(),
    }


def _print_default_results() -> None:
    results = run_default_experiments()
    print("Symmetric omitted-tail phase boundary")
    print(format_boundary_table(results["representative_boundary"]))
    print("\nFixed six-mode teacher-top-k sweep")
    print(format_comparison_table(results["ranked_mode_k"]))
    print("\nTeacher-match fixed-point check")
    print(format_teacher_match_table(results["teacher_match_k"]))
    print("\nEstimator controls (including the residual-bucket blind spot)")
    control_rows = list(results["ranked_mode_k"][:3]) + list(results["bucket_blind_spot"])
    print(format_control_comparison_table(control_rows))
    print(
        "\nIW has the exact full-RKL mean but can be noisy. The deterministic bucket is a proper coarse KL; "
        "its zero in the within-tail-swap row shows the allocation information it discards."
    )


if __name__ == "__main__":
    _print_default_results()
