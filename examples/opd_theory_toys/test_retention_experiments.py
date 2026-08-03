"""Regression tests for the exact retention/forgetting null control."""

from __future__ import annotations

import sys
from math import log
from pathlib import Path

import pytest
import torch

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from retention_experiments import (  # noqa: E402
    METHODS,
    RetentionConfig,
    context_features,
    initial_student,
    method_definitions,
    run_retention_study,
)


@pytest.fixture(scope="module")
def study():
    return run_retention_study()


def _one(study, regime: str, method: str, rho: float):
    matches = study.select(regime=regime, method=method, rho=rho)
    assert len(matches) == 1
    return matches[0]


def test_feature_geometry_and_initial_old_solution_are_exact() -> None:
    config = RetentionConfig(rhos=(0.0, 0.4, 1.0))
    student = initial_student(config)
    old, new = context_features(0.4)
    torch.testing.assert_close(old.norm(), torch.tensor(1.0, dtype=old.dtype))
    torch.testing.assert_close(new.norm(), torch.tensor(1.0, dtype=new.dtype))
    torch.testing.assert_close(torch.dot(old, new), torch.tensor(0.4, dtype=old.dtype))
    assert float(student.positive_probability(old).detach()) == pytest.approx(config.old_positive_prob, abs=1e-15)
    expected_new = torch.sigmoid(torch.tensor(0.4 * log(9.0), dtype=torch.float64))
    torch.testing.assert_close(student.positive_probability(new), expected_new)


def test_targets_utilities_and_rl_endpoint_mismatch_are_explicit() -> None:
    definitions = {row.method: row for row in method_definitions(RetentionConfig())}
    assert tuple(definitions) == METHODS
    for method in ("sft", "opd_fkl", "opd_rkl", "rl_calibrated"):
        assert definitions[method].endpoint_positive_prob == pytest.approx(0.1)
    assert definitions["rl"].utility_action_0 == 1.0
    assert definitions["rl"].utility_action_1 == 0.0
    assert definitions["rl"].endpoint_positive_prob == 0.0
    assert definitions["rl_calibrated"].reference == "uniform"
    assert definitions["rl_calibrated"].kl_beta == 1.0


def test_every_new_context_gradient_is_rank_one_and_known_equivalences_hold(study) -> None:
    for record in study.select(regime="raw_step"):
        assert record.gradient_off_feature_norm < 1e-14

    for rho in study.config.rhos:
        sft = _one(study, "raw_step", "sft", rho)
        fkl = _one(study, "raw_step", "opd_fkl", rho)
        rkl = _one(study, "raw_step", "opd_rkl", rho)
        calibrated = _one(study, "raw_step", "rl_calibrated", rho)
        assert sft.raw_gradient_norm == pytest.approx(fkl.raw_gradient_norm, abs=1e-14)
        assert rkl.raw_gradient_norm == pytest.approx(calibrated.raw_gradient_norm, abs=1e-14)


def test_zero_feature_overlap_has_exactly_zero_forgetting(study) -> None:
    for record in study.select(rho=0.0):
        assert record.step_bracketed
        assert record.old_positive_drop == 0.0
        assert record.old_target_kl_increase == 0.0
        assert record.old_behavior_kl == 0.0
        assert record.prediction_residual == 0.0


def test_raw_learning_rate_can_make_rl_look_more_retentive_only_by_learning_less(study) -> None:
    sft = _one(study, "raw_step", "sft", 1.0)
    rkl = _one(study, "raw_step", "opd_rkl", 1.0)
    rl = _one(study, "raw_step", "rl", 1.0)
    assert rl.raw_gradient_norm < rkl.raw_gradient_norm < sft.raw_gradient_norm
    assert rl.old_target_kl_increase < rkl.old_target_kl_increase < sft.old_target_kl_increase
    assert rl.new_target_gain < rkl.new_target_gain < sft.new_target_gain


def test_matching_new_task_gain_removes_the_method_label_effect(study) -> None:
    forgetting_by_overlap = []
    for rho in study.config.rhos:
        records = study.select(regime="matched_new_gain", rho=rho)
        assert len(records) == len(METHODS)
        assert all(record.step_bracketed for record in records)
        assert max(abs(record.new_target_gain - study.config.target_new_gain) for record in records) < 3e-10
        assert (
            max(record.old_positive_after for record in records) - min(record.old_positive_after for record in records)
            < 3e-10
        )
        assert max(abs(record.prediction_residual) for record in records) < 1e-14
        forgetting_by_overlap.append(records[0].old_target_kl_increase)

    assert forgetting_by_overlap[0] == 0.0
    assert all(next_value > value for value, next_value in zip(forgetting_by_overlap, forgetting_by_overlap[1:]))


def test_shared_behavior_kl_matching_also_removes_the_method_label_effect(study) -> None:
    for rho in study.config.rhos:
        records = study.select(regime="matched_behavior_kl", rho=rho)
        assert all(record.step_bracketed for record in records)
        assert max(abs(record.new_behavior_kl - study.config.target_behavior_kl) for record in records) < 2e-10
        assert (
            max(record.old_target_kl_increase for record in records)
            - min(record.old_target_kl_increase for record in records)
            < 2e-10
        )
        assert (
            max(record.new_target_gain for record in records) - min(record.new_target_gain for record in records) < 1e-9
        )
        assert max(abs(record.prediction_residual) for record in records) < 1e-14


def test_config_rejects_non_conflicting_or_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="conflict"):
        RetentionConfig(old_positive_prob=0.1, new_positive_prob=0.9)
    with pytest.raises(ValueError, match="rho"):
        RetentionConfig(rhos=(-0.1,))
    with pytest.raises(ValueError, match="rho"):
        context_features(1.1)
