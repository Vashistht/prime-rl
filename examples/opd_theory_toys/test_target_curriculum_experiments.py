"""Focused checks for the target-mismatch and curriculum experiment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from target_curriculum_experiments import (  # noqa: E402
    CategoricalPolicy,
    RunConfig,
    enumerated_grouped_rl_gradient,
    exact_population_gradients,
    informative_batch_probability,
    informative_group_probability,
    make_target_problem,
    measure_grouped_rl_no_op_probability,
    preregister_predictions,
    run_schedule,
)


def _norm(values: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def test_population_gradients_and_optima_are_preregistered_exactly() -> None:
    problem = make_target_problem()
    predictions = preregister_predictions(problem)
    gradients = predictions.initial_gradients
    p = problem.initial_probs
    q = problem.teacher_probs
    utility = problem.utility

    expected_sft_descent = q - p
    expected_rl_descent = p * (utility - (p * utility).sum())
    log_ratio = p.log() - q.log()
    expected_rkl_descent = -p * (log_ratio - (p * log_ratio).sum())
    torch.testing.assert_close(torch.tensor(gradients.sft_fkl, dtype=p.dtype), expected_sft_descent)
    torch.testing.assert_close(torch.tensor(gradients.opd_rkl, dtype=p.dtype), expected_rkl_descent)
    torch.testing.assert_close(torch.tensor(gradients.rl, dtype=p.dtype), expected_rl_descent)
    torch.testing.assert_close(
        torch.tensor(gradients.grouped_rl, dtype=p.dtype),
        0.75 * expected_rl_descent,
    )

    assert predictions.teacher_optimum == pytest.approx((0.60, 0.35, 0.05))
    assert predictions.rl_optimum == (0.0, 1.0, 0.0)
    assert predictions.teacher_expected_utility == pytest.approx(0.30)
    assert predictions.initial_expected_utility == pytest.approx(0.0)
    assert predictions.initial_expected_informative_batches == pytest.approx(1.89, rel=0.01)
    assert "current-policy analytic" in predictions.switch_rule


def test_enumerated_centered_group_gradient_has_exact_g_minus_one_over_g_scale() -> None:
    problem = make_target_problem()
    for group_size in (1, 2, 4):
        gradients = exact_population_gradients(problem, group_size=group_size)
        enumerated_loss_gradient = enumerated_grouped_rl_gradient(
            problem,
            group_size=group_size,
        )
        torch.testing.assert_close(
            -enumerated_loss_gradient,
            torch.tensor(gradients.grouped_rl, dtype=problem.initial_probs.dtype),
            atol=2e-15,
            rtol=2e-12,
        )
        assert gradients.grouped_rl_scale == pytest.approx((group_size - 1) / group_size)


def test_initial_signal_orders_match_epsilon_asymptotics() -> None:
    large_epsilon = 1e-3
    small_epsilon = 1e-6
    large = exact_population_gradients(make_target_problem(large_epsilon))
    small = exact_population_gradients(make_target_problem(small_epsilon))

    # Forward KL retains an O(1) q-p direction at the neutral vertex.
    assert _norm(large.sft_fkl) > 0.5
    assert _norm(small.sft_fkl) > 0.5
    # RL is exactly sqrt(2) * epsilon for the symmetric two-action tail.
    assert _norm(large.rl) == pytest.approx(math.sqrt(2.0) * large_epsilon)
    assert _norm(small.rl) == pytest.approx(math.sqrt(2.0) * small_epsilon)
    # Reverse KL shrinks as epsilon log(1/epsilon), up to a finite constant.
    normalized_large = _norm(large.opd_rkl) / (large_epsilon * math.log(1.0 / large_epsilon))
    normalized_small = _norm(small.opd_rkl) / (small_epsilon * math.log(1.0 / small_epsilon))
    assert normalized_small == pytest.approx(normalized_large, rel=0.20)


def test_analytic_informative_probability_predicts_empirical_no_op_rate() -> None:
    problem = make_target_problem()
    group_size, group_count = 4, 8
    group_probability = informative_group_probability(problem.initial_probs, group_size)
    batch_probability = informative_batch_probability(
        problem.initial_probs,
        group_size,
        group_count,
    )
    expected_group = 1.0 - problem.initial_probs.pow(group_size).sum()
    expected_batch = 1.0 - (1.0 - expected_group).pow(group_count)
    torch.testing.assert_close(group_probability, expected_group)
    torch.testing.assert_close(batch_probability, expected_batch)

    check = measure_grouped_rl_no_op_probability(
        problem.initial_probs,
        group_size=group_size,
        group_count=group_count,
        trials=20_000,
        seed=123,
        utility=problem.utility,
    )
    assert check.observed_batch_no_op_probability == pytest.approx(
        check.predicted_batch_no_op_probability,
        abs=4.0 * check.standard_error,
    )


def test_strict_zero_support_is_not_represented_by_finite_logits() -> None:
    with pytest.raises(ValueError, match="full support"):
        CategoricalPolicy(torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64))
    finite = CategoricalPolicy(make_target_problem(1e-12).initial_probs)
    assert bool(torch.all(finite.probs > 0))


def test_fixed_scale_view_measures_call_efficiency_and_costs() -> None:
    config = RunConfig(total_updates=30, step_mode="fixed_scale")
    pure, pure_records = run_schedule("pure_rl", seed=0, config=config)
    sft, sft_records = run_schedule("sft_then_rl", seed=0, config=config)
    opd, opd_records = run_schedule("opd_rkl_then_rl", seed=0, config=config)
    fkl, _ = run_schedule("opd_fkl_then_rl", seed=0, config=config)

    assert pure.no_op_updates == config.total_updates
    assert pure.final_expected_utility == pytest.approx(0.0)
    assert sft.switch_update == 7
    assert fkl.switch_update == 7
    assert sft.final_expected_utility > 0.85
    assert fkl.final_expected_utility > 0.85
    assert opd.switch_update is None
    assert opd.rl_updates == 0
    assert abs(opd.final_expected_utility) < 0.01

    assert sft_records[sft.switch_update - 1].informative_batch_probability >= config.switch_probability
    assert sft_records[sft.switch_update].phase == "rl"
    assert opd_records[0].applied_step_scale == pytest.approx(config.fixed_step_scale)
    assert sft_records[0].applied_step_scale == pytest.approx(config.fixed_step_scale)
    assert opd_records[0].realized_kl < 2e-6 * sft_records[0].realized_kl

    samples_per_rl_update = config.group_size * config.group_count
    assert pure.total_cost.reward_calls == config.total_updates * samples_per_rl_update
    assert sft.total_cost.teacher_hard_labels == sft.switch_update * config.acquisition_batch_size
    assert sft.total_cost.reward_calls == sft.rl_updates * samples_per_rl_update
    assert opd.total_cost.teacher_logprob_evals == config.total_updates * config.acquisition_batch_size
    assert opd.total_cost.student_rollouts == config.total_updates * config.acquisition_batch_size


def test_kl_matched_view_exposes_amplification_instead_of_call_efficiency() -> None:
    config = RunConfig(total_updates=10, step_mode="kl_matched")
    sft, sft_records = run_schedule("sft_then_rl", seed=0, config=config)
    opd, opd_records = run_schedule("opd_rkl_then_rl", seed=0, config=config)

    assert sft.switch_update == 8
    assert opd.switch_update == 8
    assert sft_records[0].step_bracketed
    assert opd_records[0].step_bracketed
    assert sft_records[0].realized_kl == pytest.approx(config.target_kl, rel=2e-3)
    assert opd_records[0].realized_kl == pytest.approx(config.target_kl, rel=2e-3)
    assert opd_records[0].raw_grad_norm < 0.01 * sft_records[0].raw_grad_norm
    assert opd_records[0].applied_step_scale > 100.0 * sft_records[0].applied_step_scale


def test_notebook_records_flatten_cost_vectors() -> None:
    summary, records = run_schedule(
        "sft_then_rl",
        seed=0,
        config=RunConfig(total_updates=2),
    )
    summary_record = summary.as_record()
    trajectory_record = records[0].as_record()
    assert "total_cost" not in summary_record
    assert "cumulative_cost" not in trajectory_record
    assert summary_record["cost_teacher_hard_labels"] == 64
    assert trajectory_record["cost_teacher_hard_labels"] == 32
