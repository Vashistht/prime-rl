from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topk_experiments import (  # noqa: E402
    enumerate_prime_gradient,
    estimator_gradient_moments,
    expected_union_gradient_prediction,
    finite_difference_prime_gradient,
    full_reverse_kl_gradient,
    ranked_mode_k_sweep,
    residual_bucket_blind_spot,
    residual_bucket_loss_and_gradient,
    symmetric_boundary_row,
    symmetric_head_tail_distributions,
    teacher_match_k_sweep,
)

DTYPE = torch.float64


@pytest.mark.parametrize(
    "proposal",
    [
        None,
        (0.12, 0.18, 0.25, 0.20, 0.15, 0.10),
    ],
)
def test_union_prediction_matches_real_prime_exact_enumeration(proposal) -> None:
    student = (0.32, 0.24, 0.17, 0.12, 0.09, 0.06)
    teacher = (0.40, 0.25, 0.14, 0.10, 0.07, 0.04)
    predicted = expected_union_gradient_prediction(student, teacher, k=2, proposal=proposal)
    enumerated = enumerate_prime_gradient(student, teacher, k=2, estimator="union", proposal=proposal)
    moments = estimator_gradient_moments(student, teacher, k=2, estimator="union", proposal=proposal)

    torch.testing.assert_close(enumerated.gradient, predicted, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(moments.mean_gradient, predicted, atol=2e-12, rtol=2e-12)


def test_importance_weighted_tail_is_exact_full_rkl_control() -> None:
    student = torch.tensor((0.37, 0.29, 0.19, 0.10, 0.05), dtype=DTYPE)
    teacher = torch.tensor((0.45, 0.25, 0.15, 0.10, 0.05), dtype=DTYPE)
    proposal = (0.10, 0.20, 0.25, 0.30, 0.15)
    result = enumerate_prime_gradient(
        student,
        teacher,
        k=2,
        estimator="importance_weighted_tail",
        proposal=proposal,
    )
    expected_loss = (student * (student.log() - teacher.log())).sum()
    expected_gradient = full_reverse_kl_gradient(student, teacher)
    moments = estimator_gradient_moments(
        student,
        teacher,
        k=2,
        estimator="importance_weighted_tail",
        proposal=proposal,
    )

    torch.testing.assert_close(result.expected_loss, expected_loss, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(result.gradient, expected_gradient, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(moments.mean_gradient, expected_gradient, atol=2e-12, rtol=2e-12)
    assert moments.variance_trace > 0.0


@pytest.mark.parametrize("estimator", ["union", "importance_weighted_tail", "residual_bucket"])
def test_real_prime_exact_surrogates_pass_finite_differences(estimator) -> None:
    check = finite_difference_prime_gradient(
        student=(0.37, 0.29, 0.19, 0.10, 0.05),
        teacher=(0.45, 0.25, 0.15, 0.10, 0.05),
        k=2,
        estimator=estimator,
    )
    assert check.max_abs_error < 2e-9


def test_symmetric_closed_form_predicts_aggregate_gradient_direction() -> None:
    mass = 0.10
    ratio = 2.0
    k = 2
    modes = 4
    student, teacher = symmetric_head_tail_distributions(mass, ratio, k=k, omitted_modes=modes)
    row = symmetric_boundary_row(mass, ratio, k=k, omitted_modes=modes, verify_prime=True)
    full_gradient = full_reverse_kl_gradient(student, teacher)
    union_gradient = expected_union_gradient_prediction(student, teacher, k=k)

    expected_full_tail_gradient = mass * (1.0 - mass) * row.full_tail_margin
    expected_union_tail_gradient = mass * (1.0 - mass) * row.union_tail_margin
    torch.testing.assert_close(
        full_gradient[k:].sum(),
        torch.tensor(expected_full_tail_gradient, dtype=DTYPE),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        union_gradient[k:].sum(),
        torch.tensor(expected_union_tail_gradient, dtype=DTYPE),
        atol=2e-12,
        rtol=2e-12,
    )
    assert row.relation == "opposes"
    assert row.prime_formula_max_error is not None and row.prime_formula_max_error < 2e-12


def test_fragmentation_flips_union_but_not_residual_bucket_direction() -> None:
    concentrated = symmetric_boundary_row(0.10, 10.0, k=2, omitted_modes=1)
    fragmented = symmetric_boundary_row(0.10, 10.0, k=2, omitted_modes=4)

    assert concentrated.full_tail_margin == pytest.approx(fragmented.full_tail_margin)
    assert concentrated.relation == "aligns"
    assert fragmented.relation == "opposes"
    assert concentrated.residual_bucket_directional_gain == pytest.approx(1.0, abs=2e-12)
    assert fragmented.residual_bucket_directional_gain == pytest.approx(1.0, abs=2e-12)
    assert fragmented.importance_weighted_relative_rms_noise > concentrated.importance_weighted_relative_rms_noise


def test_residual_bucket_is_a_proper_coarse_kl_and_prime_composition_matches() -> None:
    student = (0.37, 0.29, 0.19, 0.10, 0.05)
    teacher = (0.45, 0.25, 0.15, 0.10, 0.05)
    for k in range(1, len(student) + 1):
        loss, gradient = residual_bucket_loss_and_gradient(student, teacher, k=k)
        prime = enumerate_prime_gradient(student, teacher, k=k, estimator="residual_bucket")
        assert float(loss) >= -2e-15
        torch.testing.assert_close(prime.expected_loss, loss, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(prime.gradient, gradient, atol=2e-12, rtol=2e-12)


def test_teacher_match_is_not_union_fixed_point_but_is_bucket_fixed_point() -> None:
    rows = teacher_match_k_sweep(verify_prime=True)
    assert all(row.full_gradient_norm < 2e-14 for row in rows)
    assert all(row.union_gradient_norm > 1e-8 for row in rows[:-1])
    assert all(row.residual_bucket_gradient_norm < 2e-14 for row in rows)
    assert rows[-1].union_gradient_norm < 2e-14
    assert all(row.prime_formula_max_error is not None and row.prime_formula_max_error < 2e-12 for row in rows)
    assert all(row.residual_bucket_max_error is not None and row.residual_bucket_max_error < 2e-12 for row in rows)


def test_residual_bucket_exposes_its_within_tail_blind_spot() -> None:
    row = residual_bucket_blind_spot(verify_prime=True)
    assert row.full_gradient_norm > 0.0
    assert abs(row.residual_bucket_loss) < 2e-14
    assert row.residual_bucket_gradient_norm < 2e-14
    assert row.residual_bucket_relation == "stalled"
    assert row.residual_bucket_max_error is not None and row.residual_bucket_max_error < 2e-12
    assert row.importance_weighted_max_error is not None and row.importance_weighted_max_error < 2e-12


def test_ranked_mode_k_sweep_maps_opposition_to_full_vocab_recovery() -> None:
    rows = ranked_mode_k_sweep(verify_prime=True)
    assert [row.relation for row in rows] == ["opposes", "opposes", "opposes", "opposes", "aligns", "aligns"]
    assert rows[-1].relative_bias < 2e-12
    assert all(row.prime_formula_max_error is not None and row.prime_formula_max_error < 2e-12 for row in rows)
    assert all(
        row.importance_weighted_max_error is not None and row.importance_weighted_max_error < 2e-12 for row in rows
    )
