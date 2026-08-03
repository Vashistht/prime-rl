"""Reward-landscape and GRPO sampling checks for the truth benchmark."""

from __future__ import annotations

import sys
from itertools import product
from math import sqrt
from pathlib import Path

import pytest
import torch

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from toy_methods import RLBatch, group_advantages, rl_loss  # noqa: E402
from truth_mixture_experiments import build_decaying_problem  # noqa: E402
from truth_reward_experiments import (  # noqa: E402
    GroupSamplingConfig,
    build_mixture_reward_landscape,
    build_reward_landscape,
    conditional_informative_group_probability,
    exact_raw_update,
    expected_centered_update,
    informative_batch_probability,
    informative_group_probability,
    run_group_sampling_study,
)


def _enumerated_centered_update(landscape, current: torch.Tensor, group_size: int) -> torch.Tensor:
    """Independent finite enumeration through the shared sampled RL loss."""

    logits = current.log().clone().requires_grad_(True)
    log_probs = logits.log_softmax(dim=-1)
    expected_surrogate = logits.new_zeros(())
    n = current.numel()
    for reference in range(n):
        for action_tuple in product(range(n), repeat=group_size):
            actions = torch.tensor(action_tuple, dtype=torch.long)
            probability = landscape.truth_probs[reference] * current.gather(0, actions).prod()
            rewards = landscape.matrix[reference].gather(0, actions).unsqueeze(0)
            advantages = group_advantages(rewards, normalize=False).squeeze(0)
            conditional_loss = rl_loss(
                RLBatch(
                    student_logp=log_probs.gather(0, actions),
                    behavior_logp=current.log().gather(0, actions),
                    advantage=advantages,
                )
            ).loss
            expected_surrogate = expected_surrogate + probability * conditional_loss
    return -torch.autograd.grad(expected_surrogate, logits)[0]


@pytest.fixture(scope="module")
def truth_problem():
    return build_decaying_problem(
        mode_count=5,
        decay=0.55,
        teacher_omitted_modes=(2, 3, 4),
        omission_floor=1e-5,
        initial_student_weights=(0.82, 0.0, 0.0, 0.18, 0.0),
        support_epsilon=5e-4,
        mode_spacing=3.0,
    )


def test_reward_matrices_utilities_and_endpoints(truth_problem) -> None:
    oracle = build_mixture_reward_landscape(truth_problem, kind="oracle_log_density")
    binary = build_mixture_reward_landscape(truth_problem, kind="binary_reference")
    distance = build_mixture_reward_landscape(truth_problem, kind="squared_distance_reference")
    truth = truth_problem.truth_weights.cpu()
    indices = torch.arange(truth.numel(), dtype=torch.float64)

    torch.testing.assert_close(oracle.matrix, truth.log().expand_as(oracle.matrix))
    torch.testing.assert_close(oracle.expected_utility, truth.log())
    torch.testing.assert_close(binary.matrix, torch.eye(truth.numel(), dtype=torch.float64))
    torch.testing.assert_close(binary.expected_utility, truth)
    torch.testing.assert_close(
        distance.matrix,
        -(indices.unsqueeze(0) - indices.unsqueeze(1)).square(),
    )
    torch.testing.assert_close(distance.expected_utility, truth @ distance.matrix)

    assert oracle.maximizing_modes == (0,)
    assert binary.maximizing_modes == (0,)
    assert distance.maximizing_modes == (1,)
    assert distance.pure_endpoint.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    torch.testing.assert_close(oracle.entropy_endpoint, truth, atol=1e-14, rtol=1e-14)
    for landscape in (oracle, binary, distance):
        assert float(landscape.entropy_endpoint.sum()) == pytest.approx(1.0)
        assert bool(torch.all(landscape.entropy_endpoint > 0.0))
    assert not torch.allclose(binary.entropy_endpoint, truth, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(distance.entropy_endpoint, truth, atol=1e-4, rtol=1e-4)


def test_pure_endpoint_is_uniform_over_exact_utility_ties() -> None:
    tied = build_reward_landscape(
        (0.5, 0.5),
        (-1.0, 1.0),
        kind="binary_reference",
        spacing=2.0,
    )
    assert tied.maximizing_modes == (0, 1)
    torch.testing.assert_close(tied.pure_endpoint, torch.tensor((0.5, 0.5), dtype=torch.float64))


def test_exact_raw_update_matches_closed_form_logit_direction() -> None:
    landscape = build_reward_landscape(
        (0.5, 0.3, 0.2),
        (-1.0, 0.0, 1.0),
        kind="binary_reference",
    )
    current = torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64)
    utility = landscape.expected_utility
    expected = current * (utility - torch.dot(current, utility))
    torch.testing.assert_close(exact_raw_update(landscape, current), expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("group_size", [1, 2, 3])
def test_centered_expectation_has_exact_g_minus_one_over_g_scale(group_size: int) -> None:
    landscape = build_reward_landscape(
        (0.5, 0.3, 0.2),
        (-1.0, 0.0, 1.0),
        kind="squared_distance_reference",
    )
    current = torch.tensor((0.45, 0.35, 0.2), dtype=torch.float64)
    predicted = expected_centered_update(landscape, current, group_size)
    enumerated = _enumerated_centered_update(landscape, current, group_size)
    torch.testing.assert_close(enumerated, predicted, atol=2e-14, rtol=2e-14)
    if group_size == 1:
        torch.testing.assert_close(predicted, torch.zeros_like(predicted), atol=0.0, rtol=0.0)


def test_reward_equivalence_class_contrast_probabilities_include_distance_ties() -> None:
    truth = (0.35, 0.25, 0.2, 0.12, 0.08)
    current = torch.tensor((0.1, 0.2, 0.3, 0.15, 0.25), dtype=torch.float64)
    distance = build_reward_landscape(
        truth,
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        kind="squared_distance_reference",
    )
    binary = build_reward_landscape(
        truth,
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        kind="binary_reference",
    )
    assert distance.equivalence_classes[2] == ((0, 4), (1, 3), (2,))
    group_size = 3
    distance_homogeneous = (
        (current[0] + current[4]) ** group_size
        + (current[1] + current[3]) ** group_size
        + current[2] ** group_size
    )
    assert conditional_informative_group_probability(
        distance,
        current,
        reference=2,
        group_size=group_size,
    ) == pytest.approx(float(1.0 - distance_homogeneous), abs=1e-15)

    reference = 1
    binary_expected = 1.0 - current[reference] ** group_size - (1.0 - current[reference]) ** group_size
    assert conditional_informative_group_probability(
        binary,
        current,
        reference=reference,
        group_size=group_size,
    ) == pytest.approx(float(binary_expected), abs=1e-15)

    manual_marginal = sum(
        truth[reference]
        * conditional_informative_group_probability(
            distance,
            current,
            reference=reference,
            group_size=group_size,
        )
        for reference in range(len(truth))
    )
    group_probability = informative_group_probability(distance, current, group_size)
    assert group_probability == pytest.approx(manual_marginal, abs=1e-15)
    assert informative_batch_probability(group_probability, 7) == pytest.approx(
        1.0 - (1.0 - group_probability) ** 7,
        abs=1e-15,
    )


@pytest.fixture(scope="module")
def sampled_binary_study():
    landscape = build_reward_landscape(
        (0.50, 0.30, 0.20),
        (-1.0, 0.0, 1.0),
        kind="binary_reference",
    )
    return run_group_sampling_study(
        landscape,
        (0.45, 0.35, 0.20),
        config=GroupSamplingConfig(
            group_sizes=(1, 2, 4, 8),
            total_completions=64,
            batches_per_seed=256,
            seeds=(0, 1, 2, 3),
        ),
    )


def test_seeded_group_sampling_matches_direction_and_probability_predictions(sampled_binary_study) -> None:
    singleton = sampled_binary_study.summary(1)
    assert singleton.expected_update_norm == 0.0
    assert singleton.mean_update_norm == 0.0
    assert singleton.noise_rms == 0.0
    assert singleton.predicted_informative_group_probability == pytest.approx(0.0, abs=1e-15)
    assert singleton.observed_informative_group_probability == 0.0
    assert singleton.predicted_informative_batch_probability == 0.0
    assert singleton.observed_no_op_rate == 1.0

    for group_size in (2, 4, 8):
        summary = sampled_binary_study.summary(group_size)
        assert summary.cosine_of_mean_to_expected is not None
        assert summary.cosine_of_mean_to_expected > 0.995
        assert summary.bias_norm < 4.0 * summary.standard_error_norm
        group_trials = summary.batches * summary.groups_per_batch
        group_p = summary.predicted_informative_group_probability
        group_se = sqrt(group_p * (1.0 - group_p) / group_trials)
        assert abs(summary.observed_informative_group_probability - group_p) < 4.0 * group_se + 1e-12
        batch_p = summary.predicted_informative_batch_probability
        batch_se = sqrt(batch_p * (1.0 - batch_p) / summary.batches)
        assert abs(summary.observed_informative_batch_probability - batch_p) < 4.0 * batch_se + 1e-12
        assert summary.observed_no_op_rate + 1e-12 >= 1.0 - summary.observed_informative_batch_probability
        assert summary.snr is not None and summary.snr > 0.0


def test_invalid_reward_and_sampling_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        build_reward_landscape((0.5, 0.5, 0.0), (-1.0, 0.0, 1.0), kind="binary_reference")
    with pytest.raises(ValueError, match="strictly increasing"):
        build_reward_landscape((0.5, 0.5), (0.0, 0.0), kind="binary_reference")
    with pytest.raises(ValueError, match="spacing must be supplied"):
        build_reward_landscape((0.4, 0.3, 0.3), (0.0, 1.0, 3.0), kind="binary_reference")
    with pytest.raises(ValueError, match="unknown reward kind"):
        build_reward_landscape((0.5, 0.5), (0.0, 1.0), kind="not_a_reward")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entropy_alpha"):
        build_reward_landscape((0.5, 0.5), (0.0, 1.0), kind="binary_reference", entropy_alpha=0.0)
    with pytest.raises(ValueError, match="unique"):
        GroupSamplingConfig(group_sizes=(2, 2), total_completions=8)
    with pytest.raises(ValueError, match="divide"):
        GroupSamplingConfig(group_sizes=(3,), total_completions=8)
    with pytest.raises(ValueError, match="non-negative integers"):
        GroupSamplingConfig(group_sizes=(2,), total_completions=8, seeds=(-1,))

    landscape = build_reward_landscape((0.5, 0.5), (0.0, 1.0), kind="binary_reference")
    with pytest.raises(ValueError, match="strictly positive"):
        exact_raw_update(landscape, (1.0, 0.0))
    with pytest.raises(ValueError, match="same number"):
        run_group_sampling_study(
            landscape,
            (0.2, 0.3, 0.5),
            config=GroupSamplingConfig(group_sizes=(1,), total_completions=1, batches_per_seed=1, seeds=(0,)),
        )
