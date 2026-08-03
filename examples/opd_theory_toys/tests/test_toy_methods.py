"""Analytic invariants for the shared toy update equations."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from examples.opd_theory_toys.toy_methods import (
    FKLBatch,
    RKLBatch,
    RLBatch,
    SFTBatch,
    categorical_kl,
    categorical_kl_from_log_probs,
    categorical_metrics,
    fixed_gradient_step_,
    gradient_cosine,
    group_advantages,
    kl_matched_step_,
    loss,
    normalized_grid_log_mass,
    opd_fkl_loss,
    opd_rkl_loss,
    opd_rkl_population_loss,
    opd_rkl_population_values,
    rl_loss,
    rl_population_loss,
    rl_population_values,
    sample_categorical,
    sft_loss,
    sft_population_loss,
    sft_population_values,
    weighted_mean,
)

DTYPE = torch.float64


def _probabilities(values: list[float]) -> Tensor:
    return torch.tensor(values, dtype=DTYPE)


def _leaf_logits(probabilities: Tensor) -> Tensor:
    return probabilities.log().clone().detach().requires_grad_(True)


def test_expected_sft_and_ideal_fkl_gradients_are_p_minus_target() -> None:
    student = _probabilities([0.62, 0.28, 0.10])
    target = _probabilities([0.20, 0.50, 0.30])

    sampled_logits = _leaf_logits(student)
    sampled_logp = sampled_logits.log_softmax(dim=-1)
    expected_sft = sum(
        target[action] * sft_loss(SFTBatch(sampled_logp[action])).loss for action in range(student.numel())
    )
    sampled_gradient = torch.autograd.grad(expected_sft, sampled_logits)[0]

    exact_logits = _leaf_logits(student)
    fkl = opd_fkl_loss(FKLBatch(exact_logits.log_softmax(dim=-1), target))
    exact_gradient = torch.autograd.grad(fkl.loss, exact_logits)[0]

    expected_gradient = student - target
    torch.testing.assert_close(expected_sft, -(target * student.log()).sum())
    torch.testing.assert_close(sampled_gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(exact_gradient, expected_gradient, atol=1e-12, rtol=1e-12)


def test_sft_and_fkl_are_identical_for_the_same_population() -> None:
    logits = torch.tensor([0.8, -0.2, 0.3], dtype=DTYPE, requires_grad=True)
    demonstrations = _probabilities([0.1, 0.7, 0.2])
    sft = sft_population_loss(logits, demonstrations)
    fkl = opd_fkl_loss(FKLBatch(logits.log_softmax(dim=-1), demonstrations)).loss
    torch.testing.assert_close(sft, fkl, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        torch.autograd.grad(sft, logits, retain_graph=True)[0],
        torch.autograd.grad(fkl, logits)[0],
        atol=0.0,
        rtol=0.0,
    )


def test_vectorized_population_values_equal_individual_scalar_losses() -> None:
    logits = torch.tensor([[0.8, -0.2, 0.3], [-0.4, 1.1, 0.2]], dtype=DTYPE)
    demonstrations = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]], dtype=DTYPE)
    teacher = torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.65, 0.15]], dtype=DTYPE)
    utility = torch.tensor([[0.0, 1.0, -0.5], [1.2, -0.2, 0.4]], dtype=DTYPE)

    sft_values = sft_population_values(logits, demonstrations)
    rkl_values = opd_rkl_population_values(logits, teacher)
    rl_values = rl_population_values(logits, utility)
    assert sft_values.shape == rkl_values.shape == rl_values.shape == (2,)
    torch.testing.assert_close(sft_population_loss(logits, demonstrations), sft_values.mean())
    torch.testing.assert_close(opd_rkl_population_loss(logits, teacher), rkl_values.mean())
    torch.testing.assert_close(rl_population_loss(logits, utility), rl_values.mean())
    for row in range(logits.shape[0]):
        torch.testing.assert_close(sft_values[row], sft_population_loss(logits[row], demonstrations[row]))
        torch.testing.assert_close(rkl_values[row], opd_rkl_population_loss(logits[row], teacher[row]))
        torch.testing.assert_close(rl_values[row], rl_population_loss(logits[row], utility[row]))


def test_enumerated_prime_like_opd_is_reverse_kl_gradient() -> None:
    student = _probabilities([0.55, 0.30, 0.15])
    teacher = _probabilities([0.25, 0.50, 0.25])
    logits = _leaf_logits(student)
    logp = logits.log_softmax(dim=-1)

    per_action = torch.stack(
        [
            opd_rkl_loss(
                RKLBatch(
                    student_logp=logp[action],
                    behavior_logp=student[action].log(),
                    teacher_logp=teacher[action].log(),
                )
            ).loss
            for action in range(student.numel())
        ]
    )
    expected_surrogate = (student.detach() * per_action).sum()
    actual_gradient = torch.autograd.grad(expected_surrogate, logits)[0]

    log_ratio = student.log() - teacher.log()
    reverse_kl = (student * log_ratio).sum()
    analytic_gradient = student * (log_ratio - reverse_kl)
    torch.testing.assert_close(expected_surrogate, reverse_kl, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_gradient, analytic_gradient, atol=1e-12, rtol=1e-12)

    population_logits = _leaf_logits(student)
    population_loss = opd_rkl_population_loss(population_logits, teacher)
    population_gradient = torch.autograd.grad(population_loss, population_logits)[0]
    torch.testing.assert_close(population_loss, reverse_kl, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(population_gradient, analytic_gradient, atol=1e-12, rtol=1e-12)


def test_enumerated_rl_is_policy_gradient_and_baselines_cancel() -> None:
    policy = _probabilities([0.50, 0.35, 0.15])
    advantages = _probabilities([-0.4, 0.2, 1.3])
    logits = _leaf_logits(policy)
    logp = logits.log_softmax(dim=-1)

    per_action = torch.stack(
        [
            rl_loss(RLBatch(logp[action], policy[action].log(), advantages[action])).loss
            for action in range(policy.numel())
        ]
    )
    expected_surrogate = (policy.detach() * per_action).sum()
    actual_gradient = torch.autograd.grad(expected_surrogate, logits)[0]
    mean_advantage = (policy * advantages).sum()
    analytic_gradient = -policy * (advantages - mean_advantage)
    torch.testing.assert_close(actual_gradient, analytic_gradient, atol=1e-12, rtol=1e-12)

    baseline_logits = _leaf_logits(policy)
    baseline_logp = baseline_logits.log_softmax(dim=-1)
    constant = torch.tensor(7.0, dtype=DTYPE)
    baseline_terms = torch.stack(
        [
            rl_loss(RLBatch(baseline_logp[action], policy[action].log(), constant)).loss
            for action in range(policy.numel())
        ]
    )
    baseline_gradient = torch.autograd.grad((policy.detach() * baseline_terms).sum(), baseline_logits)[0]
    torch.testing.assert_close(baseline_gradient, torch.zeros_like(policy), atol=1e-12, rtol=0.0)


def test_losses_are_means_and_masks_do_not_change_scale() -> None:
    logp = torch.tensor([-0.2, -1.1, -0.7], dtype=DTYPE, requires_grad=True)
    original = sft_loss(SFTBatch(logp)).loss
    repeated = sft_loss(SFTBatch(logp.repeat(5))).loss
    padded = sft_loss(
        SFTBatch(
            torch.cat([logp, torch.tensor([-100.0, -100.0], dtype=DTYPE)]),
            torch.tensor([True, True, True, False, False]),
        )
    ).loss
    torch.testing.assert_close(original, repeated)
    torch.testing.assert_close(original, padded)
    all_masked = sft_loss(SFTBatch(logp, torch.zeros_like(logp, dtype=torch.bool))).loss
    all_masked_gradient = torch.autograd.grad(all_masked, logp)[0]
    torch.testing.assert_close(all_masked, torch.zeros((), dtype=DTYPE))
    torch.testing.assert_close(all_masked_gradient, torch.zeros_like(logp))


def test_weighted_mean_normalizes_masks_and_handles_zero_weight() -> None:
    values = torch.tensor([1.0, 4.0, 100.0], dtype=DTYPE, requires_grad=True)
    weights = torch.tensor([1.0, 3.0, 20.0], dtype=DTYPE, requires_grad=True)
    mask = torch.tensor([True, True, False])
    reduced = weighted_mean(values, weights, mask)
    torch.testing.assert_close(reduced, torch.tensor(3.25, dtype=DTYPE))
    gradient = torch.autograd.grad(reduced, values, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.tensor([0.25, 0.75, 0.0], dtype=DTYPE))
    assert torch.autograd.grad(reduced, weights, allow_unused=True)[0] is None

    zero_values = values.detach().clone().requires_grad_(True)
    zero = weighted_mean(zero_values, torch.zeros_like(zero_values))
    torch.testing.assert_close(zero, torch.zeros((), dtype=DTYPE))
    torch.testing.assert_close(
        torch.autograd.grad(zero, zero_values)[0],
        torch.zeros_like(zero_values),
    )


def test_weighted_mean_rejects_bad_weights() -> None:
    values = torch.ones(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="weight shape"):
        weighted_mean(values, torch.ones(2, 1, dtype=DTYPE))
    with pytest.raises(TypeError, match="floating point"):
        weighted_mean(values, torch.ones(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite and non-negative"):
        weighted_mean(values, torch.tensor([1.0, -1.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="finite and non-negative"):
        weighted_mean(values, torch.tensor([1.0, torch.nan], dtype=DTYPE))


def test_zero_teacher_mismatch_has_zero_opd_gradient() -> None:
    probabilities = _probabilities([0.2, 0.5, 0.3])
    logits = _leaf_logits(probabilities)
    logp = logits.log_softmax(dim=-1)
    losses = torch.stack(
        [opd_rkl_loss(RKLBatch(logp[i], logp.detach()[i], logp.detach()[i])).loss for i in range(probabilities.numel())]
    )
    gradient = torch.autograd.grad((probabilities * losses).sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0.0)


def test_group_advantages_center_normalize_and_detach() -> None:
    rewards = torch.tensor([[1.0, 3.0, 5.0], [2.0, 2.0, 2.0]], dtype=DTYPE, requires_grad=True)
    centered = group_advantages(rewards)
    normalized = group_advantages(rewards, normalize=True)
    torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(normalized[0].square().mean(), torch.tensor(1.0, dtype=DTYPE))
    torch.testing.assert_close(normalized[1], torch.zeros(3, dtype=DTYPE))
    assert not centered.requires_grad
    assert not normalized.requires_grad


def test_batched_categorical_sampling_is_seeded_and_records_behavior_logp() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0], [-0.5, 0.2, 0.8]], dtype=DTYPE, requires_grad=True)
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)
    first = sample_categorical(logits, 50_000, generator=first_generator)
    second = sample_categorical(logits, 50_000, generator=second_generator)

    assert first.action.shape == (2, 50_000)
    assert first.behavior_logp.shape == first.action.shape
    assert not first.behavior_logp.requires_grad
    torch.testing.assert_close(first.action, second.action)
    torch.testing.assert_close(first.behavior_logp, second.behavior_logp)
    expected_logp = logits.detach().log_softmax(dim=-1)
    torch.testing.assert_close(
        first.behavior_logp,
        expected_logp.gather(-1, first.action),
        atol=0.0,
        rtol=0.0,
    )
    frequencies = torch.stack([torch.bincount(row, minlength=3).to(DTYPE) / row.numel() for row in first.action])
    torch.testing.assert_close(frequencies, expected_logp.exp(), atol=8e-3, rtol=0.0)


def test_metrics_match_torch_kl_and_grid_matches_gaussian_kl() -> None:
    student = _probabilities([0.55, 0.30, 0.15])
    teacher = _probabilities([0.25, 0.50, 0.25])
    utility = torch.tensor([-1.0, 0.5, 2.0], dtype=DTYPE)
    metrics = categorical_metrics(
        student.log(), teacher_log_probs=teacher.log(), truth_log_probs=teacher.log(), utility=utility
    )
    torch_kl = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=student),
        torch.distributions.Categorical(probs=teacher),
    )
    torch.testing.assert_close(metrics["kl_student_teacher"], torch_kl)
    torch.testing.assert_close(metrics["expected_utility"], (student * utility).sum())
    assert float(metrics["kl_teacher_student"]) >= 0.0
    assert float(metrics["tv_teacher_student"]) >= 0.0

    grid = torch.linspace(-9.0, 9.0, 60_001, dtype=DTYPE)
    dx = float(grid[1] - grid[0])
    mean_p, std_p = -0.4, 0.8
    mean_q, std_q = 0.7, 1.3
    log_density_p = -0.5 * ((grid - mean_p) / std_p).square() - math.log(std_p * math.sqrt(2.0 * math.pi))
    log_density_q = -0.5 * ((grid - mean_q) / std_q).square() - math.log(std_q * math.sqrt(2.0 * math.pi))
    log_mass_p = normalized_grid_log_mass(log_density_p, dx=dx)
    log_mass_q = normalized_grid_log_mass(log_density_q, dx=dx)
    grid_kl = categorical_kl(log_mass_p.exp(), log_mass_q.exp())
    analytic_kl = math.log(std_q / std_p) + (std_p**2 + (mean_p - mean_q) ** 2) / (2.0 * std_q**2) - 0.5
    torch.testing.assert_close(grid_kl, torch.tensor(analytic_kl, dtype=DTYPE), atol=2e-8, rtol=0.0)


def test_log_space_kl_is_batched_and_does_not_invent_zero_support() -> None:
    # exp(-1000) underflows in float64 although the normalized log mass is
    # finite.  KL must remain finite unless the log mass is exactly -inf.
    log_p = torch.tensor([[0.0, -1000.0], [-1000.0, 0.0]], dtype=DTYPE).log_softmax(dim=-1)
    log_q = torch.tensor([[-1000.0, 0.0], [0.0, -1000.0]], dtype=DTYPE).log_softmax(dim=-1)
    assert bool(torch.all(log_q.exp().amin(dim=-1) == 0.0))

    values = categorical_kl_from_log_probs(log_p, log_q)
    assert values.shape == (2,)
    assert bool(torch.all(torch.isfinite(values)))
    torch.testing.assert_close(values, torch.full((2,), 1000.0, dtype=DTYPE))

    metrics = categorical_metrics(
        log_q[0],
        teacher_log_probs=log_p[0],
        truth_log_probs=log_p[0],
    )
    assert math.isfinite(float(metrics["kl_teacher_student"]))
    assert math.isfinite(float(metrics["kl_student_teacher"]))
    torch.testing.assert_close(metrics["kl_teacher_student"], values[0])


def test_log_space_kl_preserves_genuine_negative_infinity_support() -> None:
    log_half = math.log(0.5)
    log_p = torch.tensor([log_half, log_half, -torch.inf], dtype=DTYPE)
    log_q = torch.tensor([log_half, -torch.inf, log_half], dtype=DTYPE)
    assert bool(torch.isinf(categorical_kl_from_log_probs(log_p, log_q)))

    same_support = torch.tensor([0.0, -torch.inf], dtype=DTYPE)
    torch.testing.assert_close(
        categorical_kl_from_log_probs(same_support, same_support),
        torch.zeros((), dtype=DTYPE),
    )


def test_population_rkl_uses_stable_log_student_probabilities() -> None:
    logits = torch.tensor([[0.0, -1000.0], [-1000.0, 0.0]], dtype=DTYPE, requires_grad=True)
    teacher = torch.full_like(logits, 0.5)
    values = opd_rkl_population_values(logits, teacher)
    expected = categorical_kl_from_log_probs(logits.log_softmax(dim=-1), teacher.log())
    torch.testing.assert_close(values, expected)
    assert bool(torch.all(torch.isfinite(values)))
    gradient = torch.autograd.grad(values.sum(), logits)[0]
    assert bool(torch.all(torch.isfinite(gradient)))


class _LogitPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.4, -0.3, 0.1], dtype=DTYPE))


@pytest.mark.parametrize("max_grad_norm", [None, 0.1])
def test_fixed_gradient_step_preserves_raw_scale_until_cap(max_grad_norm: float | None) -> None:
    model = _LogitPolicy()
    original = model.logits.detach().clone()
    old_logp = original.log_softmax(dim=-1)
    old_p = old_logp.exp()
    objective = -model.logits.log_softmax(dim=-1)[2]
    gradient = torch.autograd.grad(objective, model.logits, retain_graph=True)[0]

    report = fixed_gradient_step_(
        model,
        objective,
        lambda: (old_p * (old_logp - model.logits.log_softmax(dim=-1))).sum(),
        step_scale=0.4,
        max_grad_norm=max_grad_norm,
    )
    expected_clip = max_grad_norm is not None and float(gradient.norm()) > max_grad_norm
    assert report.clipped is expected_clip
    expected_scale = 0.4 if not expected_clip else 0.4 * max_grad_norm / float(gradient.norm())
    assert report.applied_scale == pytest.approx(expected_scale)
    torch.testing.assert_close(model.logits - original, -expected_scale * gradient)
    assert report.realized_kl > 0.0
    assert not report.no_op


def test_fixed_gradient_step_reports_exact_no_op() -> None:
    model = _LogitPolicy()
    original = model.logits.detach().clone()
    objective = (0.0 * model.logits).sum()
    report = fixed_gradient_step_(model, objective, lambda: torch.zeros((), dtype=DTYPE), step_scale=1.0)
    assert report.no_op
    assert report.grad_norm == 0.0
    assert report.applied_scale == 0.0
    torch.testing.assert_close(model.logits, original, atol=0.0, rtol=0.0)


def test_failed_fixed_step_metric_restores_snapshot() -> None:
    model = _LogitPolicy()
    original = model.logits.detach().clone()
    objective = -model.logits.log_softmax(dim=-1)[2]

    def broken_closure() -> Tensor:
        raise RuntimeError("intentional fixed-step metric failure")

    with pytest.raises(RuntimeError, match="intentional fixed-step"):
        fixed_gradient_step_(model, objective, broken_closure, step_scale=1.0)
    torch.testing.assert_close(model.logits, original, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("initial_scale", [1e-5, 10.0])
def test_kl_matched_step_hits_target_and_preserves_direction(initial_scale: float) -> None:
    model = _LogitPolicy()
    original = model.logits.detach().clone()
    old_logp = original.log_softmax(dim=-1)
    old_p = old_logp.exp()
    objective = -model.logits.log_softmax(dim=-1)[2]
    gradient = torch.autograd.grad(objective, model.logits, retain_graph=True)[0]

    def old_to_new_kl() -> Tensor:
        current_logp = model.logits.log_softmax(dim=-1)
        return (old_p * (old_logp - current_logp)).sum()

    report = kl_matched_step_(
        model,
        objective,
        old_to_new_kl,
        target_kl=0.01,
        initial_scale=initial_scale,
        max_scale=100.0,
        rtol=1e-5,
    )
    assert report.bracketed
    assert report.realized_kl == pytest.approx(0.01, rel=1e-5)
    torch.testing.assert_close(model.logits - original, -report.scale * gradient)
    cosine = gradient_cosine(model.logits - original, -gradient)
    assert float(cosine.detach()) == pytest.approx(1.0)


def test_unbracketed_or_failed_kl_search_restores_snapshot() -> None:
    model = _LogitPolicy()
    original = model.logits.detach().clone()
    old_logp = original.log_softmax(dim=-1)
    old_p = old_logp.exp()
    objective = -model.logits.log_softmax(dim=-1)[2]

    report = kl_matched_step_(
        model,
        objective,
        lambda: (old_p * (old_logp - model.logits.log_softmax(dim=-1))).sum(),
        target_kl=0.5,
        initial_scale=1e-8,
        max_scale=1e-8,
    )
    assert not report.bracketed
    assert report.scale == 0.0
    torch.testing.assert_close(model.logits, original, atol=0.0, rtol=0.0)

    failed_model = _LogitPolicy()
    failed_original = failed_model.logits.detach().clone()
    failed_objective = -failed_model.logits.log_softmax(dim=-1)[2]

    def broken_closure() -> Tensor:
        raise RuntimeError("intentional probe failure")

    with pytest.raises(RuntimeError, match="intentional"):
        kl_matched_step_(
            failed_model,
            failed_objective,
            broken_closure,
            target_kl=0.01,
        )
    torch.testing.assert_close(failed_model.logits, failed_original, atol=0.0, rtol=0.0)


def test_dispatch_rejects_crossed_batch_types() -> None:
    logp = torch.tensor(-0.7, dtype=DTYPE, requires_grad=True)
    assert loss("sft", SFTBatch(logp)).loss.ndim == 0
    with pytest.raises(TypeError, match="expects RKLBatch"):
        loss("opd_rkl", SFTBatch(logp))
    with pytest.raises(ValueError, match="unknown method"):
        loss("not_a_method", SFTBatch(logp))  # type: ignore[arg-type]
