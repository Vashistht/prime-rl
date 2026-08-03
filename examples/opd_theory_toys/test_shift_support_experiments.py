"""Regression tests for the first-principles shifted-support experiment."""

from __future__ import annotations

import sys
from math import exp
from pathlib import Path

import pytest
import torch

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from shift_support_experiments import (  # noqa: E402
    ShiftConfig,
    bracketing_batch_sizes,
    compact_update_rows,
    discovery_probability,
    effective_batch_boundary,
    fixed_width_student,
    flat_interval_teacher,
    gaussian_tail_teacher,
    idealized_bridge_plan,
    ordered_grid,
    run_shift_sweep,
    signal_predictions,
    target_mask,
    verify_discovery_model,
)


@pytest.fixture(scope="module")
def config() -> ShiftConfig:
    return ShiftConfig(grid_step=0.1, target_kl=5e-4)


def _exact(result, teacher: str, method: str, displacement: float, epsilon=None):
    matches = [
        record
        for record in result.exact_updates
        if record.teacher_shape == teacher
        and record.method == method
        and record.displacement == displacement
        and record.epsilon == epsilon
    ]
    assert len(matches) == 1
    return matches[0]


def test_ordered_grid_teacher_shapes_and_fixed_width_student(config: ShiftConfig) -> None:
    grid = ordered_grid(config)
    student = fixed_width_student(config)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    assert trainable == [student.mean]
    assert bool(torch.all(torch.diff(grid.points) > 0.0))

    center = 3.5
    mask = target_mask(grid, center, config.interval_width)
    gaussian = gaussian_tail_teacher(grid, center, config.teacher_std)
    flat = flat_interval_teacher(grid, center, config.interval_width, 1e-3)
    torch.testing.assert_close(gaussian.sum(), torch.tensor(1.0, dtype=gaussian.dtype))
    torch.testing.assert_close(flat.sum(), torch.tensor(1.0, dtype=flat.dtype))
    assert bool(torch.all(gaussian > 0.0))
    assert bool(torch.all(flat > 0.0))

    # The B teacher is flat in density (mass divided by cell width) off target.
    off_target_density = (flat / grid.weights)[~mask]
    torch.testing.assert_close(
        off_target_density,
        off_target_density[0].expand_as(off_target_density),
        atol=1e-15,
        rtol=1e-12,
    )
    assert float((flat / grid.weights)[mask].min()) > float(off_target_density[0])


def test_population_signals_match_preregistered_model_and_kl_matching(config: ShiftConfig) -> None:
    result = run_shift_sweep(
        displacements=(1.5, 3.5),
        epsilons=(1e-3,),
        batch_sizes=(64,),
        seeds=(0,),
        config=config,
        include_bridges=False,
    )
    predictions = {(row.teacher_shape, row.displacement): row for row in result.predictions}

    for displacement in (1.5, 3.5):
        for teacher in ("gaussian_tail", "flat_interval"):
            prediction = predictions[(teacher, displacement)]
            opd = _exact(result, teacher, "opd_rkl", displacement, None if teacher == "gaussian_tail" else 1e-3)
            sft = _exact(result, teacher, "sft", displacement, None if teacher == "gaussian_tail" else 1e-3)
            fkl = _exact(result, teacher, "opd_fkl", displacement, None if teacher == "gaussian_tail" else 1e-3)
            assert opd.raw_toward_signal == pytest.approx(prediction.opd_toward_signal, rel=2e-9, abs=1e-11)
            assert sft.raw_toward_signal == pytest.approx(prediction.sft_toward_signal, rel=2e-9, abs=1e-11)
            assert fkl.raw_toward_signal == pytest.approx(sft.raw_toward_signal, abs=1e-12)
            assert fkl.mean_delta_toward == pytest.approx(sft.mean_delta_toward, abs=1e-12)

        rl = _exact(result, "sparse_reward", "rl", displacement)
        prediction = predictions[("gaussian_tail", displacement)]
        assert rl.raw_toward_signal == pytest.approx(prediction.sparse_rl_toward_signal, abs=1e-12)

    # Gaussian log tails retain an O(displacement) direction.  Flat OPD and
    # sparse RL inherit the exponentially shrinking interval-overlap derivative.
    gaussian_far = _exact(result, "gaussian_tail", "opd_rkl", 3.5)
    flat_far = _exact(result, "flat_interval", "opd_rkl", 3.5, 1e-3)
    rl_near = _exact(result, "sparse_reward", "rl", 1.5)
    rl_far = _exact(result, "sparse_reward", "rl", 3.5)
    assert gaussian_far.raw_toward_signal > 50.0 * flat_far.raw_toward_signal
    assert rl_far.raw_toward_signal < 0.05 * rl_near.raw_toward_signal

    # Conditional on a nonzero correct direction, behavioral KL matching makes
    # step geometry comparable without erasing the raw-signal distinction.
    for record in result.exact_updates:
        assert record.step_bracketed
        assert record.realized_kl == pytest.approx(config.target_kl, rel=3e-4)
        assert record.mean_delta_toward == pytest.approx((2.0 * config.target_kl) ** 0.5, rel=2e-4)


def test_epsilon_only_changes_flat_opd_signal_logarithmically(config: ShiftConfig) -> None:
    grid = ordered_grid(config)
    _, loose = signal_predictions(3.5, 1e-1, config=config, grid=grid)
    _, tiny = signal_predictions(3.5, 1e-6, config=config, grid=grid)
    assert tiny.flat_log_density_contrast > loose.flat_log_density_contrast
    assert tiny.opd_toward_signal > loose.opd_toward_signal
    assert tiny.opd_toward_signal < 5.0 * loose.opd_toward_signal
    assert tiny.sparse_rl_toward_signal == pytest.approx(loose.sparse_rl_toward_signal)
    assert tiny.opd_toward_signal < 0.05 * 3.5  # still far below Gaussian-tail OPD


def test_discovery_formula_boundary_and_seeded_monte_carlo(config: ShiftConfig) -> None:
    chi = 1e-4
    boundary = effective_batch_boundary(chi)
    assert boundary == pytest.approx(10_000.0)
    assert discovery_probability(chi, round(boundary)) == pytest.approx(1.0 - exp(-1.0), abs=5e-5)
    assert discovery_probability(chi, 100) < 0.011
    assert discovery_probability(chi, 100_000) > 0.9999
    lower, upper = bracketing_batch_sizes(chi)
    assert lower * chi <= 0.5
    assert upper * chi >= 1.0

    records = verify_discovery_model(
        2.5,
        batch_sizes=(16, 64, 256),
        seeds=tuple(range(512)),
        config=config,
    )
    for record in records:
        tolerance = 4.0 * record.standard_error + 1.0 / record.trials
        assert abs(record.observed_probability - record.predicted_probability) <= tolerance
        assert record.n_chi == pytest.approx(record.batch_size * record.chi)


def test_zero_hit_rows_separate_tail_information_score_noise_and_zero_gradient(
    config: ShiftConfig,
) -> None:
    result = run_shift_sweep(
        displacements=(4.5,),
        epsilons=(1e-3,),
        batch_sizes=(64,),
        seeds=tuple(range(16)),
        config=config,
        include_bridges=False,
    )
    sampled = result.sampled_updates
    rl = [row for row in sampled if row.method == "rl"]
    gaussian_opd = [
        row for row in sampled if row.method == "opd_rkl" and row.teacher_shape == "gaussian_tail"
    ]
    flat_opd = [
        row for row in sampled if row.method == "opd_rkl" and row.teacher_shape == "flat_interval"
    ]
    assert all(row.target_hits == 0 for row in rl + gaussian_opd + flat_opd)
    assert all(row.sample_regime == "zero_gradient_no_hit" for row in rl)
    assert all(row.grad_norm == 0.0 and not row.step_bracketed for row in rl)
    assert all(row.sample_regime == "tail_signal_no_hit" for row in gaussian_opd)
    assert sum(row.raw_toward_signal > 0.0 for row in gaussian_opd) >= 14
    assert all(row.sample_regime == "score_noise_no_hit" for row in flat_opd)
    assert all(row.grad_norm > 0.0 for row in flat_opd)

    compact = compact_update_rows(result)
    sampled_rl = next(row for row in compact if row["method"] == "rl" and row["estimator"] == "sampled")
    sampled_flat = next(
        row
        for row in compact
        if row["method"] == "opd_rkl"
        and row["teacher"] == "flat_interval"
        and row["estimator"] == "sampled"
    )
    assert sampled_rl["sample_regimes"] == "zero_gradient_no_hit:16"
    assert sampled_flat["sample_regimes"] == "score_noise_no_hit:16"


def test_bridge_plan_is_only_the_n_chi_reachability_calculation(config: ShiftConfig) -> None:
    hard = idealized_bridge_plan(4.5, 64, config=config)
    direct = idealized_bridge_plan(4.5, 32_768, config=config)
    assert hard.direct_n_chi < 1.0
    assert hard.stage_count is not None and hard.stage_count > 1
    assert hard.stage_centers[-1] == pytest.approx(config.student_mean + 4.5)
    increments = [
        current - previous
        for previous, current in zip((config.student_mean, *hard.stage_centers[:-1]), hard.stage_centers)
    ]
    assert all(increment <= hard.max_discoverable_stage_shift + 1e-12 for increment in increments)
    assert direct.direct_n_chi >= 1.0
    assert direct.stage_count == 1
    assert direct.stage_centers == pytest.approx((config.student_mean + 4.5,))
