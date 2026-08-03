"""Small categorical checks against Prime-RL's real SFT, OPD, and RL losses.

This module intentionally contains no copied loss implementations.  It calls
the current Prime-RL loss functions on one-token categorical examples, then
compares their gradients with analytic population gradients.  Everything is
CPU-only and uses float64 so that discrepancies are easy to distinguish from
round-off.

Run from an environment in which this checkout of ``prime_rl`` is importable::

    python examples/opd_theory_toys/prime_loss_checks.py

The top-k experiment enumerates every possible rollout token.  Its reported
gradient is therefore the *expected minibatch gradient*, not a noisy Monte
Carlo estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch
from torch import Tensor

try:
    from .toy_methods import (
        opd_rkl_population_loss,
        rl_population_loss,
        sft_population_loss,
    )
except ImportError:  # pragma: no cover - supports direct ``python path/to/file.py``
    from toy_methods import (
        opd_rkl_population_loss,
        rl_population_loss,
        sft_population_loss,
    )

try:
    from prime_rl.configs.trainer import DefaultLossConfig
    from prime_rl.trainer.rl.loss import LossInputs, default_loss_fn, opd_loss_fn, sft_loss_fn
except ImportError as error:  # pragma: no cover - only reached outside a Prime-RL runtime
    raise ImportError(
        "Could not import the current Prime-RL losses. Run this file with the "
        "existing Prime-RL environment (or install this checkout editable)."
    ) from error


DTYPE = torch.float64
_LOSS_INPUT_FIELD_NAMES = {field.name for field in fields(LossInputs)}
PRIME_HAS_TOPK_OPD = {
    "teacher_topk_logprobs",
    "student_topk_logprobs",
    "sampled_token_in_teacher_topk",
}.issubset(_LOSS_INPUT_FIELD_NAMES)


@dataclass(frozen=True)
class GradientCheck:
    """One implementation gradient and the population quantity it should equal."""

    name: str
    actual_loss: Tensor
    expected_loss: Tensor
    actual_gradient: Tensor
    expected_gradient: Tensor

    @property
    def loss_error(self) -> float:
        return float((self.actual_loss - self.expected_loss).abs())

    @property
    def gradient_max_error(self) -> float:
        return float((self.actual_gradient - self.expected_gradient).abs().max())


@dataclass(frozen=True)
class TopKBiasRow:
    """Expected sparse-OPD gradient bias for one rare-tail probability."""

    student_tail_probability: float
    teacher_tail_probability: float
    student_topk_mass: float
    full_gradient_norm: float
    sparse_gradient_norm: float
    bias_norm: float
    relative_bias: float
    cosine_similarity: float
    directional_gain: float


def _probabilities(values: Sequence[float]) -> Tensor:
    probabilities = torch.tensor(values, dtype=DTYPE)
    if probabilities.ndim != 1 or probabilities.numel() < 2:
        raise ValueError("A categorical distribution must be a one-dimensional vector of length >= 2.")
    if not bool(torch.all(probabilities > 0)):
        raise ValueError("All probabilities must be strictly positive.")
    if not torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=DTYPE), atol=1e-12, rtol=1e-12):
        raise ValueError(f"Probabilities must sum to one; got {float(probabilities.sum()):.16g}.")
    return probabilities


def _logits_for(probabilities: Tensor) -> Tensor:
    """Return leaf logits whose softmax is exactly ``probabilities`` up to round-off."""

    return probabilities.log().clone().detach().requires_grad_(True)


def _one_token_inputs(
    student_logprob: Tensor,
    *,
    inference_logprob: Tensor | None = None,
    teacher_logprob: Tensor | None = None,
    advantage: float = 0.0,
    teacher_topk_logprobs: Tensor | None = None,
    student_topk_logprobs: Tensor | None = None,
    sampled_token_in_teacher_topk: bool | None = None,
) -> LossInputs:
    """Build the one-token shapes expected by Prime-RL's ``LossInputs``."""

    if inference_logprob is None:
        inference_logprob = student_logprob.detach()
    membership = None
    if sampled_token_in_teacher_topk is not None:
        membership = torch.tensor([sampled_token_in_teacher_topk], dtype=torch.bool)
    kwargs = dict(
        trainer_logprobs=student_logprob.reshape(1),
        inference_logprobs=inference_logprob.reshape(1),
        teacher_logprobs=None if teacher_logprob is None else teacher_logprob.reshape(1),
        advantages=torch.tensor([advantage], dtype=DTYPE),
        loss_mask=torch.ones(1, dtype=torch.bool),
    )
    topk_requested = any(
        value is not None
        for value in (
            teacher_topk_logprobs,
            student_topk_logprobs,
            sampled_token_in_teacher_topk,
        )
    )
    if topk_requested and not PRIME_HAS_TOPK_OPD:
        raise RuntimeError(
            "This Prime-RL installation predates teacher-top-k-plus-sampled OPD. "
            "Use the current checkout/runtime for sparse top-k experiments."
        )
    if PRIME_HAS_TOPK_OPD:
        kwargs.update(
            teacher_topk_logprobs=(None if teacher_topk_logprobs is None else teacher_topk_logprobs.reshape(1, -1)),
            student_topk_logprobs=(None if student_topk_logprobs is None else student_topk_logprobs.reshape(1, -1)),
            sampled_token_in_teacher_topk=membership,
        )
    return LossInputs(**kwargs)


def categorical_expected_sft_check(
    student: Sequence[float] = (0.62, 0.28, 0.10),
    demonstrations: Sequence[float] = (0.20, 0.50, 0.30),
) -> GradientCheck:
    """Check that expected Prime-RL SFT is cross-entropy with gradient ``p - q``.

    We enumerate the demonstration token ``y ~ q`` and weight the real
    one-token SFT loss by ``q[y]``.  This removes sampling noise and exposes
    the forward-KL / support-covering population update.
    """

    student_p = _probabilities(student)
    demo_p = _probabilities(demonstrations)
    logits = _logits_for(student_p)
    student_logp = logits.log_softmax(dim=-1)

    expected_sft_loss = torch.zeros((), dtype=DTYPE)
    for token in range(student_p.numel()):
        token_loss = sft_loss_fn(_one_token_inputs(student_logp[token])).loss
        expected_sft_loss = expected_sft_loss + demo_p[token] * token_loss
    actual_gradient = torch.autograd.grad(expected_sft_loss, logits)[0]

    # The expected side comes from the shared toy operator.  This module only
    # owns the enumeration adapter around Prime-RL's real implementation.
    expected_logits = _logits_for(student_p)
    shared_loss = sft_population_loss(expected_logits, demo_p)
    shared_gradient = torch.autograd.grad(shared_loss, expected_logits)[0]
    return GradientCheck(
        name="expected SFT",
        actual_loss=expected_sft_loss.detach(),
        expected_loss=shared_loss.detach(),
        actual_gradient=actual_gradient.detach(),
        expected_gradient=shared_gradient.detach(),
    )


def _analytic_reverse_kl(student_p: Tensor, teacher_p: Tensor) -> tuple[Tensor, Tensor]:
    """Compatibility adapter used by the top-k experiments.

    The population value and gradient deliberately come from
    :mod:`toy_methods`; keeping this private name stable avoids coupling the
    top-k implementation experiment to the refactor.
    """

    logits = _logits_for(student_p)
    reverse_kl = opd_rkl_population_loss(logits, teacher_p)
    gradient = torch.autograd.grad(reverse_kl, logits)[0]
    return reverse_kl.detach(), gradient.detach()


def full_vocab_opd_rkl_checks(
    student: Sequence[float] = (0.55, 0.30, 0.15),
    teacher: Sequence[float] = (0.25, 0.50, 0.25),
) -> dict[str, GradientCheck]:
    """Check both current Prime-RL routes to a full-vocabulary reverse KL.

    ``full-vocab sparse route`` gives ``opd_loss_fn`` a teacher top-k whose
    ``k`` equals the vocabulary size, making its sparse sum exact.

    ``expected sampled route`` uses the ordinary one-sample OPD path and
    enumerates an on-policy rollout ``y ~ p``.  Its per-sample loss is a
    score-function surrogate; its expectation has the same reverse-KL
    gradient even though a single sample does not.
    """

    student_p = _probabilities(student)
    teacher_p = _probabilities(teacher)
    analytic_loss, analytic_gradient = _analytic_reverse_kl(student_p, teacher_p)
    vocab_size = student_p.numel()

    teacher_logp = teacher_p.log()
    checks: dict[str, GradientCheck] = {}
    if PRIME_HAS_TOPK_OPD:
        # Exact full-vocabulary sum through the newer top-k OPD API.
        direct_logits = _logits_for(student_p)
        direct_logp = direct_logits.log_softmax(dim=-1)
        all_teacher_ids = teacher_logp.topk(vocab_size).indices
        sampled_token = 0
        direct_output = opd_loss_fn(
            _one_token_inputs(
                direct_logp[sampled_token],
                teacher_logprob=teacher_logp[sampled_token],
                teacher_topk_logprobs=teacher_logp[all_teacher_ids],
                student_topk_logprobs=direct_logp[all_teacher_ids],
                sampled_token_in_teacher_topk=True,
            )
        )
        direct_gradient = torch.autograd.grad(direct_output.loss, direct_logits)[0]
        checks["full-vocab sparse route"] = GradientCheck(
            name="full-vocab OPD sparse route",
            actual_loss=direct_output.loss.detach(),
            expected_loss=analytic_loss,
            actual_gradient=direct_gradient.detach(),
            expected_gradient=analytic_gradient,
        )

    # Expected gradient of the ordinary sampled, on-policy OPD route.
    sampled_logits = _logits_for(student_p)
    sampled_logp = sampled_logits.log_softmax(dim=-1)
    expected_sampled_loss = torch.zeros((), dtype=DTYPE)
    for token in range(vocab_size):
        token_output = opd_loss_fn(
            _one_token_inputs(
                sampled_logp[token],
                inference_logprob=sampled_logp[token].detach(),
                teacher_logprob=teacher_logp[token],
            )
        )
        # Rollout probabilities are data, hence detached from the loss graph.
        expected_sampled_loss = expected_sampled_loss + student_p[token] * token_output.loss
    sampled_gradient = torch.autograd.grad(expected_sampled_loss, sampled_logits)[0]

    checks["expected sampled route"] = GradientCheck(
        name="expected on-policy OPD sampled route",
        actual_loss=expected_sampled_loss.detach(),
        expected_loss=analytic_loss,
        actual_gradient=sampled_gradient.detach(),
        expected_gradient=analytic_gradient,
    )
    return checks


def on_policy_rl_gradient_check(
    policy: Sequence[float] = (0.50, 0.35, 0.15),
    advantages: Sequence[float] = (-0.4, 0.2, 1.3),
) -> GradientCheck:
    """Check Prime-RL's on-policy RL gradient against the policy-gradient identity."""

    policy_p = _probabilities(policy)
    advantage = torch.tensor(advantages, dtype=DTYPE)
    if advantage.shape != policy_p.shape:
        raise ValueError("There must be one advantage per categorical action.")

    logits = _logits_for(policy_p)
    logp = logits.log_softmax(dim=-1)
    expected_rl_loss = torch.zeros((), dtype=DTYPE)
    loss_config = DefaultLossConfig(
        adv_tau=1.0,
        kl_tau=0.0,
        dppo_mask_low=1.0,
        dppo_mask_high=1.0,
    )
    for action in range(policy_p.numel()):
        action_output = default_loss_fn(
            _one_token_inputs(
                logp[action],
                inference_logprob=logp[action].detach(),
                advantage=float(advantage[action]),
            ),
            loss_config,
        )
        expected_rl_loss = expected_rl_loss + policy_p[action] * action_output.loss
    actual_gradient = torch.autograd.grad(expected_rl_loss, logits)[0]

    expected_logits = _logits_for(policy_p)
    shared_loss = rl_population_loss(expected_logits, advantage)
    shared_gradient = torch.autograd.grad(shared_loss, expected_logits)[0]
    return GradientCheck(
        name="expected on-policy RL",
        actual_loss=expected_rl_loss.detach(),
        expected_loss=shared_loss.detach(),
        actual_gradient=actual_gradient.detach(),
        expected_gradient=shared_gradient.detach(),
    )


def _expected_topk_union_gradient(
    student_p: Tensor,
    teacher_p: Tensor,
    *,
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exactly enumerate the gradient produced by top-k-union-sampled OPD."""

    if not PRIME_HAS_TOPK_OPD:
        raise RuntimeError("teacher-top-k-plus-sampled OPD is unavailable in this Prime-RL installation")
    if not 1 <= k <= student_p.numel():
        raise ValueError(f"k must be in [1, {student_p.numel()}], got {k}.")
    logits = _logits_for(student_p)
    student_logp = logits.log_softmax(dim=-1)
    teacher_logp = teacher_p.log()
    teacher_topk_ids = teacher_logp.topk(k).indices
    expected_sparse_loss = torch.zeros((), dtype=DTYPE)

    for sampled_token in range(student_p.numel()):
        sampled_in_topk = bool(torch.any(teacher_topk_ids == sampled_token))
        output = opd_loss_fn(
            _one_token_inputs(
                student_logp[sampled_token],
                inference_logprob=student_logp[sampled_token].detach(),
                teacher_logprob=teacher_logp[sampled_token],
                teacher_topk_logprobs=teacher_logp[teacher_topk_ids],
                student_topk_logprobs=student_logp[teacher_topk_ids],
                sampled_token_in_teacher_topk=sampled_in_topk,
            )
        )
        # This is expectation over fixed rollout data, not differentiation
        # through the sampling distribution.
        expected_sparse_loss = expected_sparse_loss + student_p[sampled_token] * output.loss

    gradient = torch.autograd.grad(expected_sparse_loss, logits)[0]
    return expected_sparse_loss.detach(), gradient.detach(), teacher_topk_ids


def teacher_topk_union_bias_sweep(
    tail_probabilities: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    *,
    student_to_teacher_tail_ratio: float = 10.0,
    k: int = 2,
) -> list[TopKBiasRow]:
    """Measure sparse OPD gradient bias as a mismatched tail becomes rarer.

    The vocabulary has two head tokens and one tail token.  Student and
    teacher agree on the conditional 65/35 split *within* the head, while the
    student puts ``student_to_teacher_tail_ratio`` times more mass on the tail.
    Thus the experiment isolates missing tail mass rather than an unrelated
    head-ranking error.  Teacher top-2 excludes the tail throughout the
    default sweep.

    ``directional_gain`` is ``<g_sparse, g_full> / ||g_full||^2``.  A value of
    one is unbiased, zero omits the useful direction, and a negative value
    points against the full reverse-KL gradient.
    """

    if not PRIME_HAS_TOPK_OPD:
        raise RuntimeError("teacher-top-k-plus-sampled OPD is unavailable in this Prime-RL installation")
    if student_to_teacher_tail_ratio <= 1.0:
        raise ValueError("Use a ratio > 1 to study an over-weighted student tail.")
    head_conditional = torch.tensor([0.65, 0.35], dtype=DTYPE)
    rows: list[TopKBiasRow] = []

    for tail_probability in tail_probabilities:
        if not 0.0 < tail_probability < 1.0:
            raise ValueError(f"Tail probabilities must lie in (0, 1); got {tail_probability}.")
        teacher_tail_probability = tail_probability / student_to_teacher_tail_ratio
        student_p = torch.cat(
            [(1.0 - tail_probability) * head_conditional, torch.tensor([tail_probability], dtype=DTYPE)]
        )
        teacher_p = torch.cat(
            [
                (1.0 - teacher_tail_probability) * head_conditional,
                torch.tensor([teacher_tail_probability], dtype=DTYPE),
            ]
        )
        _, sparse_gradient, teacher_topk_ids = _expected_topk_union_gradient(student_p, teacher_p, k=k)
        _, full_gradient = _analytic_reverse_kl(student_p, teacher_p)

        if bool(torch.any(teacher_topk_ids == student_p.numel() - 1)):
            raise ValueError(
                "The requested sweep no longer leaves the tail outside teacher top-k; "
                "reduce k or the largest tail probability."
            )

        bias = sparse_gradient - full_gradient
        full_norm = torch.linalg.vector_norm(full_gradient)
        sparse_norm = torch.linalg.vector_norm(sparse_gradient)
        bias_norm = torch.linalg.vector_norm(bias)
        cosine = torch.dot(sparse_gradient, full_gradient) / (sparse_norm * full_norm)
        directional_gain = torch.dot(sparse_gradient, full_gradient) / full_norm.square()
        student_topk_mass = student_p[teacher_topk_ids].sum()
        rows.append(
            TopKBiasRow(
                student_tail_probability=float(tail_probability),
                teacher_tail_probability=float(teacher_tail_probability),
                student_topk_mass=float(student_topk_mass),
                full_gradient_norm=float(full_norm),
                sparse_gradient_norm=float(sparse_norm),
                bias_norm=float(bias_norm),
                relative_bias=float(bias_norm / full_norm),
                cosine_similarity=float(cosine),
                directional_gain=float(directional_gain),
            )
        )

    return rows


def _assert_gradient_check(check: GradientCheck, *, atol: float = 2e-12) -> None:
    torch.testing.assert_close(check.actual_loss, check.expected_loss, atol=atol, rtol=atol)
    torch.testing.assert_close(check.actual_gradient, check.expected_gradient, atol=atol, rtol=atol)


def test_categorical_expected_sft() -> None:
    _assert_gradient_check(categorical_expected_sft_check())


def test_full_vocab_opd_is_reverse_kl() -> None:
    for check in full_vocab_opd_rkl_checks().values():
        _assert_gradient_check(check)


def test_on_policy_rl_gradient() -> None:
    _assert_gradient_check(on_policy_rl_gradient_check())


def test_teacher_topk_union_sampled_bias_is_exactly_enumerated() -> None:
    if not PRIME_HAS_TOPK_OPD:
        import pytest

        pytest.skip("installed Prime-RL predates teacher-top-k-plus-sampled OPD")
    rows = teacher_topk_union_bias_sweep()
    assert len(rows) == 8
    assert all(row.bias_norm > 0.0 for row in rows)
    # In this controlled missing-tail example, the sparse expected gradient
    # points against the true RKL gradient.  This is the phenomenon the sweep
    # is designed to expose, rather than Monte Carlo noise.
    assert all(row.cosine_similarity < -0.999999 for row in rows)
    assert all(row.directional_gain < 0.0 for row in rows)


def run_checks() -> tuple[list[GradientCheck], list[TopKBiasRow]]:
    """Run all deterministic checks and return their inspectable results."""

    checks = [categorical_expected_sft_check()]
    checks.extend(full_vocab_opd_rkl_checks().values())
    checks.append(on_policy_rl_gradient_check())
    for check in checks:
        _assert_gradient_check(check)
    bias_rows = teacher_topk_union_bias_sweep() if PRIME_HAS_TOPK_OPD else []
    if PRIME_HAS_TOPK_OPD:
        test_teacher_topk_union_sampled_bias_is_exactly_enumerated()
    return checks, bias_rows


def _print_results(checks: Sequence[GradientCheck], bias_rows: Sequence[TopKBiasRow]) -> None:
    print("Prime-RL categorical loss checks (float64, CPU)")
    for check in checks:
        print(
            f"  PASS  {check.name:<39} loss error={check.loss_error:.3e}  grad max error={check.gradient_max_error:.3e}"
        )

    if not bias_rows:
        print("  SKIP   sparse top-k checks (not exposed by this Prime-RL installation)")
        return

    print("\nExpected teacher-top-2 ∪ sampled-token OPD gradient bias")
    print("  student tail  teacher tail  top-k mass   ||g_full||   rel. bias   cosine    dir. gain")
    for row in bias_rows:
        print(
            f"  {row.student_tail_probability:12.1e}  "
            f"{row.teacher_tail_probability:12.1e}  "
            f"{row.student_topk_mass:10.6f}  "
            f"{row.full_gradient_norm:10.3e}  "
            f"{row.relative_bias:9.3f}  "
            f"{row.cosine_similarity:7.3f}  "
            f"{row.directional_gain:9.3f}"
        )
    print(
        "\nInterpretation: in this isolated missing-tail construction, sparse OPD's expected "
        "gradient has negative directional gain even though the sampled token is unioned into "
        "teacher top-k. Rare samples expose the omitted token only with probability p_tail, so "
        "the union term does not reproduce the full-vocabulary RKL gradient in expectation."
    )


if __name__ == "__main__":
    _print_results(*run_checks())
