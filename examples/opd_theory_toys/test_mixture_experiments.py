"""Fast regression checks for the asymmetric-mixture experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TOY_DIRECTORY = Path(__file__).resolve().parent
if str(TOY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOY_DIRECTORY))

from families import GaussianPolicy1D, QuadratureGrid1D  # noqa: E402
from mixture_experiments import (  # noqa: E402
    LandscapeConfig,
    asymmetric_teacher,
    build_objective_landscape,
    numerical_entropy_switch,
    pure_rl_collapse_scan,
    theory_predictions,
)
from toy_methods import SFTBatch, sft_loss  # noqa: E402


@pytest.fixture(scope="module")
def small_landscape():
    teacher = asymmetric_teacher()
    grid = QuadratureGrid1D.composite(
        lower=-15.0,
        dense_lower=-7.0,
        dense_upper=6.0,
        upper=15.0,
        tail_step=0.1,
        dense_step=0.02,
    )
    config = LandscapeConfig(
        mean_min=-5.0,
        mean_max=5.0,
        mean_count=101,
        min_std=0.04,
        max_std=4.0,
        log_std_count=81,
        chunk_size=128,
    )
    return teacher, grid, build_objective_landscape(teacher, grid=grid, config=config)


def test_theory_predictions_are_fixed_by_moments_and_component_scales() -> None:
    predictions = theory_predictions()
    assert predictions.sft_mean == pytest.approx(-2.4)
    assert predictions.sft_variance == pytest.approx(11.042)
    assert predictions.sft_std == pytest.approx(3.322950496170534)
    assert predictions.right_peak_log_density > predictions.left_peak_log_density
    assert predictions.pure_rl_mode == "right"
    assert predictions.opd_global_mode == "left"
    assert predictions.entropy_switch_alpha == pytest.approx(0.3979400086720376)


def test_sft_moment_projection_is_stationary_under_shared_loss() -> None:
    """Two equal cubature points exactly reproduce the Gaussian sufficient statistics."""

    teacher = asymmetric_teacher()
    policy = GaussianPolicy1D(float(teacher.mean), float(teacher.variance.sqrt()))
    cubature = teacher.mean + teacher.variance.sqrt() * torch.tensor(
        (-1.0, 1.0), dtype=teacher.means.dtype
    )
    loss = sft_loss(SFTBatch(policy.log_prob(cubature))).loss
    mean_gradient, log_std_gradient = torch.autograd.grad(loss, tuple(policy.parameters()))
    assert float(mean_gradient.abs()) < 1e-12
    assert float(log_std_gradient.abs()) < 1e-12


def test_global_landscape_separates_sft_opd_and_rl(small_landscape) -> None:
    _, _, landscape = small_landscape
    sft = landscape.minimum("sft")
    opd = landscape.minimum("opd")
    pure_rl = landscape.minimum("rl")

    assert sft.mean == pytest.approx(-2.4, abs=0.11)
    assert sft.std == pytest.approx(3.32295, abs=0.2)
    assert opd.mean == pytest.approx(-4.0, abs=0.11)
    assert opd.std == pytest.approx(1.0, abs=0.08)
    assert pure_rl.mean == pytest.approx(4.0, abs=0.11)
    assert pure_rl.log_std_index == 0
    assert pure_rl.on_search_boundary


def test_numerical_entropy_switch_matches_prediction(small_landscape) -> None:
    _, _, landscape = small_landscape
    assert numerical_entropy_switch(landscape) == pytest.approx(0.39794, abs=2e-3)
    assert landscape.minimum("entropy_rl", alpha=0.35).mean > 0.0
    assert landscape.minimum("entropy_rl", alpha=0.5).mean < 0.0
    alpha_one = landscape.minimum("entropy_rl", alpha=1.0)
    opd = landscape.minimum("opd")
    assert alpha_one.mean == opd.mean
    assert alpha_one.std == opd.std


def test_pure_rl_has_no_finite_variance_optimum(small_landscape) -> None:
    teacher, grid, _ = small_landscape
    scan = pure_rl_collapse_scan((0.3, 0.2, 0.1, 0.05), teacher=teacher, grid=grid)
    losses = [loss for _, loss in scan]
    assert all(next_loss < loss for loss, next_loss in zip(losses, losses[1:]))
