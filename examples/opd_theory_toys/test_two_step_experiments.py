"""Invariants for the horizon-two state-occupancy experiment."""

from __future__ import annotations

import math

import torch

from examples.opd_theory_toys.two_step_experiments import (
    COLLECTORS,
    DISTILLATION_METHODS,
    exact_population_records,
    finite_sample_records,
    make_two_step_fork,
    predictions,
    sparse_action_rl_records,
    sparse_rl_prediction,
    summarize_finite_records,
)


def _find(records, *, collector: str, method: str):
    return next(record for record in records if record.collector == collector and record.method == method)


def test_fork_is_a_genuine_frozen_occupancy_mismatch() -> None:
    low = make_two_step_fork(0.0)
    high = make_two_step_fork(1.0)
    torch.testing.assert_close(low.teacher_occupancy, torch.tensor([1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(low.student_occupancy, torch.tensor([0.2, 0.8], dtype=torch.float64))
    assert low.teacher_conditionals.shape == (2, 3)
    assert not torch.equal(low.teacher_conditionals[1], low.initial_conditionals[1])
    torch.testing.assert_close(low.teacher_conditionals[1, 1], torch.tensor(0.05, dtype=torch.float64))
    torch.testing.assert_close(high.teacher_conditionals[1, 1], torch.tensor(0.90, dtype=torch.float64))
    torch.testing.assert_close(low.teacher_conditionals[:, 2], torch.tensor([0.05, 0.05], dtype=torch.float64))


def test_exact_factorial_separates_coverage_from_projection() -> None:
    records = exact_population_records((0.75,), target_kl=0.005)
    assert len(records) == len(COLLECTORS) * len(DISTILLATION_METHODS)
    problem = make_two_step_fork(0.75)

    for method in DISTILLATION_METHODS:
        teacher = _find(records, collector="teacher", method=method)
        student = _find(records, collector="student", method=method)
        torch.testing.assert_close(
            torch.tensor(teacher.recovery_update, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.tensor(teacher.canonical_recovery_probs, dtype=torch.float64),
            problem.initial_conditionals[1],
        )
        torch.testing.assert_close(
            torch.tensor(teacher.matched_recovery_probs, dtype=torch.float64),
            problem.initial_conditionals[1],
        )
        assert teacher.identified_states == (True, False)
        assert student.identified_states == (True, True)
        assert torch.linalg.vector_norm(torch.tensor(student.recovery_update, dtype=torch.float64)) > 0
        torch.testing.assert_close(
            torch.tensor(student.canonical_recovery_probs, dtype=torch.float64),
            problem.teacher_conditionals[1],
        )
        assert teacher.matched_step_bracketed and student.matched_step_bracketed

    sft = _find(records, collector="student", method="sft_hard")
    fkl = _find(records, collector="student", method="opd_fkl")
    torch.testing.assert_close(
        torch.tensor(sft.common_update, dtype=torch.float64),
        torch.tensor(fkl.common_update, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.tensor(sft.recovery_update, dtype=torch.float64),
        torch.tensor(fkl.recovery_update, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert not torch.allclose(
        torch.tensor(fkl.recovery_update, dtype=torch.float64),
        torch.tensor(_find(records, collector="student", method="opd_rkl").recovery_update, dtype=torch.float64),
    )


def test_reliability_thresholds_are_fixed_before_sampling() -> None:
    result = predictions()
    assert result.top_action_agreement_threshold == 0.5
    assert math.isclose(result.recovery_improvement_threshold, 3.0 / 17.0)
    initial = result.initial_recovery_trusted_probability
    assert float(make_two_step_fork(0.1).teacher_conditionals[1, 1]) < initial
    assert float(make_two_step_fork(0.3).teacher_conditionals[1, 1]) > initial

    # Endpoint harm does not imply an initially harmful step in this
    # three-action construction: every projection first removes the much
    # larger distractor mass.  This explicitly checks the distinction made in
    # ``Predictions`` under the common KL-matched step budget.
    unreliable_records = exact_population_records((0.0,), target_kl=0.01)
    for method in DISTILLATION_METHODS:
        record = _find(unreliable_records, collector="student", method=method)
        assert record.canonical_recovery_probs[1] < initial
        assert record.matched_recovery_probs[1] > initial


def test_finite_factorial_is_reproducible_and_preserves_structural_noncoverage() -> None:
    kwargs = dict(reliabilities=(0.0, 1.0), query_counts=(4, 12), seeds=(3, 7), target_kl=0.005)
    first = finite_sample_records(**kwargs)
    second = finite_sample_records(**kwargs)
    assert first == second
    assert len(first) == 2 * len(COLLECTORS) * len(DISTILLATION_METHODS) * 2 * 2

    for record in first:
        if record.collector == "teacher":
            assert record.recovery_queries == 0
            assert record.recovery_trusted_probability == 0.2
        else:
            assert 0 <= record.recovery_queries <= record.query_count

    summaries = summarize_finite_records(first)
    assert len(summaries) == 2 * len(COLLECTORS) * len(DISTILLATION_METHODS) * 2
    for summary in summaries:
        if summary.collector == "teacher":
            assert summary.recovery_seen_rate == 0.0
            assert summary.mean_recovery_trusted_probability == 0.2


def test_sparse_rl_prediction_matches_signal_event_and_updates_only_when_informative() -> None:
    p = 0.01
    prediction = sparse_rl_prediction(p, group_size=4, group_count=3)
    homogeneous = (1.0 - p) ** 4 + p**4
    assert math.isclose(prediction.group_informative_probability, 1.0 - homogeneous)
    assert math.isclose(prediction.batch_informative_probability, 1.0 - homogeneous**3)

    records = sparse_action_rl_records(
        action_count=8,
        initial_jackpot_probability=p,
        group_counts=(1, 16),
        group_size=4,
        seeds=tuple(range(16)),
        target_kl=0.005,
    )
    assert any(record.informative_groups == 0 for record in records)
    assert any(record.informative_groups > 0 for record in records)
    for record in records:
        assert record.step_bracketed == (record.informative_groups > 0)
        if record.informative_groups == 0:
            assert math.isclose(record.jackpot_probability_after_update, p, rel_tol=0.0, abs_tol=1e-15)
            assert record.realized_kl == 0.0
        else:
            assert record.jackpot_probability_after_update > p
            assert math.isclose(record.realized_kl, 0.005, rel_tol=6e-3)
