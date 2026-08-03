"""Regression tests for finite sampling in the named-mode truth benchmark."""

from __future__ import annotations

import sys
from math import isinf, sqrt
from pathlib import Path

import pytest

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from truth_sampling_experiments import (  # noqa: E402
    METHODS,
    SamplingConfig,
    discovery_probability,
    discovery_rows,
    exact_rows,
    run_truth_sampling_study,
    summary_rows,
)

TRUTH = (0.50, 0.30, 0.15, 0.05)
TEACHER = (0.55, 0.35, 0.099, 0.001)
STUDENT = (0.05, 0.10, 0.25, 0.60)


@pytest.fixture(scope="module")
def study():
    return run_truth_sampling_study(
        TRUTH,
        TEACHER,
        STUDENT,
        config=SamplingConfig(batch_sizes=(8, 128), seeds=tuple(range(96))),
    )


def _discovery(study, source: str, batch_size: int, mode: int):
    matches = [
        row for row in study.discovery if row.source == source and row.batch_size == batch_size and row.mode == mode
    ]
    assert len(matches) == 1
    return matches[0]


def test_exact_gradients_use_the_expected_population_objectives(study) -> None:
    sft = study.exact("sft_hard")
    fkl = study.exact("opd_fkl_full")
    entropy_rl = study.exact("rl_entropy_truth")
    pure_rl = study.exact("rl_pure_truth")
    validity = study.exact("rl_constant_validity")

    assert sft.gradient == pytest.approx(fkl.gradient, abs=1e-15)
    assert sft.objective_value == pytest.approx(fkl.objective_value, abs=1e-15)
    assert entropy_rl.gradient != pytest.approx(pure_rl.gradient, abs=1e-8)
    assert validity.gradient_norm == 0.0
    assert validity.gradient == pytest.approx((0.0,) * len(TRUTH), abs=1e-15)
    for record in study.exact_gradients:
        assert sum(record.gradient) == pytest.approx(0.0, abs=2e-15)

    # With a truthful teacher, entropy-calibrated truth RL and reverse-KL PD
    # are the same population gradient.  This uses another shared-loss run,
    # rather than an analytic gradient copied into the test.
    truthful = run_truth_sampling_study(
        TRUTH,
        TRUTH,
        STUDENT,
        config=SamplingConfig(batch_sizes=(1,), seeds=(0,)),
    )
    assert truthful.exact("opd_rkl").gradient == pytest.approx(
        truthful.exact("rl_entropy_truth").gradient,
        abs=2e-15,
    )


def test_gradient_estimators_converge_to_exact_shared_gradients(study) -> None:
    stochastic_methods = (
        "sft_hard",
        "opd_rkl",
        "rl_pure_truth",
        "rl_entropy_truth",
        "rl_constant_validity",
    )
    for method in stochastic_methods:
        small = study.summary(method, 8)
        large = study.summary(method, 128)
        # The deterministic seed set should lie comfortably inside its own
        # Monte Carlo standard-error scale.
        assert large.bias_norm < 4.0 * large.standard_error_norm
        assert large.noise_rms < small.noise_rms
        if method == "rl_constant_validity":
            assert large.snr == 0.0
            assert large.cosine_of_mean_to_exact is None
        else:
            assert large.snr is not None and large.snr > small.snr
            assert large.cosine_of_mean_to_exact is not None
            assert large.cosine_of_mean_to_exact > 0.99


def test_full_logit_fkl_is_a_deterministic_zero_variance_control(study) -> None:
    for batch_size in study.config.batch_sizes:
        summary = study.summary("opd_fkl_full", batch_size)
        assert summary.std_gradient == (0.0,) * len(TRUTH)
        assert summary.noise_rms == 0.0
        assert summary.std_loss == 0.0
        assert summary.bias_norm < 2e-15
        assert summary.snr is not None and isinf(summary.snr)


def test_discovery_predictions_match_expected_counts_and_observed_hit_rates(study) -> None:
    for record in study.discovery:
        assert record.expected_count == pytest.approx(record.batch_size * record.mode_probability)
        assert record.predicted_hit_probability == pytest.approx(
            discovery_probability(record.mode_probability, record.batch_size),
            abs=1e-15,
        )
        count_standard_error = sqrt(
            record.batch_size * record.mode_probability * (1.0 - record.mode_probability) / record.trials
        )
        assert abs(record.observed_mean_count - record.expected_count) <= 4.0 * count_standard_error + 1e-12
        assert abs(record.observed_hit_rate - record.predicted_hit_probability) <= (
            4.0 * record.predicted_hit_rate_standard_error + 1.0 / record.trials
        )

    # Teacher SFT almost never observes its epsilon mode at N=8, while the
    # student-sampled methods see that same named mode because the student
    # currently concentrates there.
    teacher_rare = _discovery(study, "teacher", 8, 3)
    student_mode = _discovery(study, "student", 8, 3)
    assert teacher_rare.expected_count == pytest.approx(0.008)
    assert teacher_rare.predicted_hit_probability < 0.01
    assert student_mode.predicted_hit_probability > 0.99


def test_constant_validity_reward_cannot_identify_truth_weights() -> None:
    config = SamplingConfig(batch_sizes=(1,), seeds=(0,))
    first = run_truth_sampling_study(TRUTH, TEACHER, STUDENT, config=config)
    second = run_truth_sampling_study((0.1, 0.2, 0.3, 0.4), TEACHER, STUDENT, config=config)
    assert first.exact("rl_constant_validity").gradient == second.exact("rl_constant_validity").gradient
    assert first.exact("rl_constant_validity").objective_value == pytest.approx(-1.0)
    assert second.exact("rl_constant_validity").objective_value == pytest.approx(-1.0)
    assert first.exact("rl_pure_truth").gradient != pytest.approx(
        second.exact("rl_pure_truth").gradient,
        abs=1e-8,
    )


def test_records_are_notebook_ready(study) -> None:
    assert len(exact_rows(study)) == len(METHODS)
    assert len(summary_rows(study)) == len(METHODS) * len(study.config.batch_sizes)
    assert len(discovery_rows(study)) == 2 * len(study.config.batch_sizes) * len(TRUTH)
    assert set(summary_rows(study)[0]) >= {"method", "mean_gradient", "snr", "cosine_of_mean_to_exact"}


def test_inputs_must_be_strict_positive_normalized_vectors() -> None:
    config = SamplingConfig(batch_sizes=(1,), seeds=(0,))
    with pytest.raises(ValueError, match="strictly positive"):
        run_truth_sampling_study((0.5, 0.5, 0.0), (0.5, 0.4, 0.1), (0.3, 0.3, 0.4), config=config)
    with pytest.raises(ValueError, match="sum to one"):
        run_truth_sampling_study((0.5, 0.4), (0.5, 0.5), (0.5, 0.5), config=config)
    with pytest.raises(ValueError, match="same number"):
        run_truth_sampling_study((0.5, 0.5), (0.5, 0.5), (0.2, 0.3, 0.5), config=config)
    with pytest.raises(ValueError, match="unique"):
        SamplingConfig(batch_sizes=(8, 8), seeds=(0,))
