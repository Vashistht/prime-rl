"""Invariants and endpoints for the decaying truth-mixture benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from truth_mixture_experiments import (  # noqa: E402
    METHODS,
    ModePolicy,
    TrajectoryConfig,
    UndefinedPopulationObjective,
    build_decaying_problem,
    exact_initial_location_gradient,
    population_loss,
    render_mixtures,
    run_labeled_location_trajectory,
    run_population_study,
    run_population_trajectory,
    theory_predictions,
)


def test_builder_exposes_decay_distortion_omission_and_geometry() -> None:
    problem = build_decaying_problem(
        mode_count=4,
        decay=0.5,
        teacher_omitted_modes=(2,),
        omission_floor=1e-5,
        teacher_weight_power=0.5,
        initial_student_modes=(1,),
        support_epsilon=1e-3,
        global_shift=2.0,
        component_shifts=(0.0, 0.1, -0.2, 0.3),
        teacher_global_shift=0.7,
        teacher_component_shifts=(0.0, 0.2, 0.0, -0.1),
        student_global_shift=-0.4,
        student_component_shifts=(0.1, 0.0, -0.2, 0.0),
    )
    ratios = problem.truth_weights[1:] / problem.truth_weights[:-1]
    torch.testing.assert_close(ratios, torch.full_like(ratios, 0.5))
    assert problem.omitted_teacher_modes == (2,)
    assert 0.0 < float(problem.teacher_weights[2]) < 1e-4
    assert int(problem.initial_student_weights.argmax()) == 1
    assert bool(torch.all(problem.initial_student_weights > 0))
    assert problem.initial_student_weights[0] == pytest.approx(1e-3)
    torch.testing.assert_close(
        problem.teacher_mode_means - problem.truth_mode_means,
        problem.truth_mode_means.new_tensor((0.7, 0.9, 0.7, 0.6)),
    )
    torch.testing.assert_close(
        problem.student_mode_means - problem.truth_mode_means,
        problem.truth_mode_means.new_tensor((-0.3, -0.4, -0.6, -0.4)),
    )


def test_sft_and_full_fkl_are_the_same_population_objective_and_gradient() -> None:
    problem = build_decaying_problem(mode_count=4, teacher_omitted_modes=(3,))
    sft_policy = ModePolicy(problem)
    fkl_policy = ModePolicy(problem)
    sft = population_loss("sft", sft_policy, problem)
    fkl = population_loss("opd_fkl", fkl_policy, problem)
    sft_gradient = torch.autograd.grad(sft, sft_policy.logits)[0]
    fkl_gradient = torch.autograd.grad(fkl, fkl_policy.logits)[0]
    torch.testing.assert_close(sft, fkl, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(sft_gradient, fkl_gradient, atol=1e-14, rtol=1e-14)


def test_entropy_calibration_and_pure_rl_endpoints_follow_truth() -> None:
    problem = build_decaying_problem(mode_count=5)
    policy = ModePolicy(problem)
    calibrated = population_loss("truth_rl_entropy", policy, problem, entropy_alpha=1.0)
    truth = problem.truth_weights
    student = policy.full_probs
    expected_kl = (student * (student.log() - truth.log())).sum()
    torch.testing.assert_close(calibrated, expected_kl, atol=1e-14, rtol=1e-14)

    predictions = theory_predictions(problem)
    assert predictions.pure_truth_rl_selected_mode == 0
    assert predictions.pure_truth_rl_endpoint == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert predictions.entropy_truth_rl_endpoint == pytest.approx(tuple(float(x) for x in truth))
    assert predictions.omitted_truth_weight == pytest.approx(float(truth[-1]))


def test_decay_is_strict_and_small_temperature_prediction_is_stable() -> None:
    with pytest.raises(ValueError, match="largest truth mode is unique"):
        build_decaying_problem(decay=1.0)

    problem = build_decaying_problem(mode_count=5, decay=0.55)
    alpha = 1e-3
    predictions = theory_predictions(problem, entropy_alpha=alpha)
    predicted = torch.tensor(
        predictions.entropy_truth_rl_endpoint,
        dtype=problem.truth_weights.dtype,
    )
    expected = (problem.truth_weights.log() / alpha).softmax(dim=-1)
    assert bool(torch.all(torch.isfinite(predicted)))
    torch.testing.assert_close(predicted, expected, atol=0.0, rtol=0.0)
    assert float(predicted.sum()) == pytest.approx(1.0)


def test_exact_teacher_omission_is_not_silently_smoothed_for_reverse_kl() -> None:
    full_capacity = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(3,),
        omission_floor=0.0,
    )
    predictions = theory_predictions(full_capacity)
    assert not predictions.opd_rkl_available
    with pytest.raises(UndefinedPopulationObjective, match="reverse KL is infinite"):
        run_population_trajectory(
            "opd_rkl",
            problem=full_capacity,
            config=TrajectoryConfig(steps=2),
        )

    matched_capacity = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(3,),
        omission_floor=0.0,
        student_capacity_modes=(0, 1, 2),
    )
    matched_predictions = theory_predictions(matched_capacity)
    assert matched_predictions.opd_rkl_available
    assert matched_predictions.sft_fkl_available
    trajectory = run_population_trajectory(
        "opd_rkl",
        problem=matched_capacity,
        config=TrajectoryConfig(steps=5, record_every=5),
    )
    assert trajectory.final.metrics.student_weights[3] == 0.0


def test_unlisted_zero_teacher_multiplier_is_recorded_as_omission() -> None:
    problem = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(),
        teacher_weight_multipliers=(1.0, 0.0, 1.0, 1.0),
    )
    assert problem.omitted_teacher_modes == (1,)
    predictions = theory_predictions(problem)
    assert not predictions.opd_rkl_available
    assert predictions.omitted_truth_weight == pytest.approx(float(problem.truth_weights[1]))


def test_epsilon_initialization_is_recoverable_but_capacity_mask_is_structural() -> None:
    epsilon_problem = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(),
        initial_student_modes=(0,),
        support_epsilon=1e-4,
    )
    assert bool(torch.all(epsilon_problem.initial_student_weights > 0))
    epsilon_trajectory = run_population_trajectory(
        "truth_rl_entropy",
        problem=epsilon_problem,
        config=TrajectoryConfig(steps=40, learning_rate=0.2, record_every=40),
    )
    assert epsilon_trajectory.final.metrics.student_weights[3] > 0.5 * float(
        epsilon_problem.truth_weights[3]
    )

    masked_problem = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(2, 3),
        omission_floor=0.0,
        initial_student_modes=(0,),
        student_capacity_modes=(0, 1),
    )
    masked = run_population_trajectory(
        "truth_rl_entropy",
        problem=masked_problem,
        config=TrajectoryConfig(steps=20, record_every=20),
    )
    assert masked.final.metrics.student_weights[2:] == (0.0, 0.0)
    assert masked.final.metrics.kl_truth_student == torch.inf
    expected_recall = float(masked_problem.truth_weights[:2].sum())
    assert masked.final.metrics.truth_support_recall == pytest.approx(expected_recall)


@pytest.fixture(scope="module")
def default_study():
    return run_population_study(
        config=TrajectoryConfig(steps=40, learning_rate=0.2, record_every=10)
    )


def test_deterministic_population_endpoints_and_forward_path_identity(default_study) -> None:
    sft = default_study.trajectory("sft")
    fkl = default_study.trajectory("opd_fkl")
    rkl = default_study.trajectory("opd_rkl")
    pure = default_study.trajectory("truth_rl")
    calibrated = default_study.trajectory("truth_rl_entropy")

    assert sft.points == fkl.points
    assert sft.final.metrics.kl_teacher_student < 3e-3
    assert rkl.final.metrics.kl_student_teacher < 1e-2
    assert pure.final.metrics.student_weights[0] > 0.99
    assert calibrated.final.metrics.kl_truth_student < 1e-3
    assert sft.final.metrics.student_omitted_teacher_mass < 3e-3
    assert calibrated.final.metrics.student_omitted_teacher_mass == pytest.approx(
        default_study.predictions.omitted_truth_weight,
        abs=4e-3,
    )


def test_render_shifts_are_explicit_but_leave_core_weight_loss_invariant() -> None:
    base = build_decaying_problem(mode_count=3, teacher_omitted_modes=(), global_shift=0.0)
    translated = build_decaying_problem(mode_count=3, teacher_omitted_modes=(), global_shift=10.0)
    base_loss = population_loss("opd_rkl", ModePolicy(base), base)
    translated_loss = population_loss("opd_rkl", ModePolicy(translated), translated)
    torch.testing.assert_close(base_loss, translated_loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        translated.truth_mode_means - base.truth_mode_means,
        torch.full_like(base.truth_mode_means, 10.0),
    )
    rendered = render_mixtures(translated)
    assert rendered.points.ndim == 1
    assert rendered.truth_density.shape == rendered.points.shape
    assert bool(torch.all(rendered.truth_density >= 0))


def test_labeled_location_gradient_has_the_predicted_weighting_and_recovers_shift() -> None:
    problem = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(),
        teacher_global_shift=0.7,
        student_global_shift=-0.4,
    )
    variance = problem.component_std**2
    sft = exact_initial_location_gradient("sft", problem)
    fkl = exact_initial_location_gradient("opd_fkl", problem)
    rkl = exact_initial_location_gradient("opd_rkl", problem)
    pure = exact_initial_location_gradient("truth_rl", problem)
    calibrated = exact_initial_location_gradient("truth_rl_entropy", problem)
    assert sft.weighting_distribution == "teacher"
    assert rkl.weighting_distribution == "student"
    assert sft.gradient == pytest.approx(-1.1 / variance)
    assert fkl.gradient == pytest.approx(sft.gradient)
    assert rkl.gradient == pytest.approx(-1.1 / variance)
    assert pure.gradient == pytest.approx(-0.4 / variance)
    assert calibrated.gradient == pytest.approx(pure.gradient)
    assert sum(sft.component_gradients) == pytest.approx(sft.gradient)
    assert sum(rkl.component_gradients) == pytest.approx(rkl.gradient)
    assert sum(pure.component_gradients) == pytest.approx(pure.gradient)

    trajectory = run_labeled_location_trajectory(
        "sft",
        problem=problem,
        config=TrajectoryConfig(steps=50, learning_rate=0.1, record_every=10),
    )
    assert trajectory.final.metrics.translation == pytest.approx(1.1, abs=0.05)
    record = trajectory.records()[-1]
    assert "student_weights" in record
    assert "teacher_weighted_teacher_location_rmse" in record

    mode_specific = build_decaying_problem(
        mode_count=4,
        teacher_omitted_modes=(),
        initial_student_modes=(0,),
        support_epsilon=1e-3,
        teacher_component_shifts=(0.0, 1.0, 0.0, 0.0),
    )
    forward_specific = exact_initial_location_gradient("opd_fkl", mode_specific)
    reverse_specific = exact_initial_location_gradient("opd_rkl", mode_specific)
    displaced_mode = 1
    displacement = float(
        mode_specific.teacher_mode_means[displaced_mode]
        - mode_specific.student_mode_means[displaced_mode]
    )
    expected_forward = (
        -float(mode_specific.teacher_weights[displaced_mode])
        * displacement
        / mode_specific.component_std**2
    )
    expected_reverse = (
        -float(mode_specific.initial_student_weights[displaced_mode])
        * displacement
        / mode_specific.component_std**2
    )
    assert forward_specific.component_gradients[displaced_mode] == pytest.approx(expected_forward)
    assert reverse_specific.component_gradients[displaced_mode] == pytest.approx(expected_reverse)


def test_notebook_records_are_flat_and_complete(default_study) -> None:
    record = default_study.records()[0]
    expected = {
        "method",
        "step",
        "loss",
        "student_weights",
        "entropy",
        "effective_support",
        "expected_truth_log_weight",
        "kl_truth_student",
        "kl_student_truth",
        "kl_teacher_student",
        "kl_student_teacher",
        "truth_support_recall",
        "student_omitted_teacher_mass",
        "largest_student_mode",
    }
    assert expected == set(record)
    assert {trajectory.method for trajectory in default_study.trajectories} == set(METHODS)
