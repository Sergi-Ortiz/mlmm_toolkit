"""Two runs of the same calculation draw the same numbers.

fairchem reseeds Python, NumPy and torch every time a predictor is built,
so the random numbers in a run (UMA's reference vector, the lobpcg and
Lanczos start vectors) depend only on the sequence of calls. That sequence
is the same in two runs only if every kernel is deterministic.
"""
import os
import random

import numpy as np
import pytest
import torch

from mlmm.mlmm_calc import seed_random_streams, use_deterministic_torch


@pytest.fixture
def restore_torch_settings():
    """Undo the process-wide settings a test changes."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if cublas_config is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config


def test_deterministic_torch_turns_on_every_switch(restore_torch_settings):
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    use_deterministic_torch()

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_torch_keeps_a_cublas_config_already_set(restore_torch_settings):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    use_deterministic_torch()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def draw_from_every_stream():
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_same_seed_gives_the_same_draws():
    seed_random_streams(7)
    first = draw_from_every_stream()
    seed_random_streams(7)
    second = draw_from_every_stream()

    assert first == second


def test_different_seeds_give_different_draws():
    seed_random_streams(7)
    first = draw_from_every_stream()
    seed_random_streams(8)
    second = draw_from_every_stream()

    assert all(a != b for a, b in zip(first, second))


def test_the_pysisyphus_calculator_passes_both_options_to_the_ml_core(recording_core):
    import mlmm.mlmm_pysis as mlmm_pysis

    calc = mlmm_pysis.mlmm(model_charge=0, deterministic=False, seed=5)

    assert calc.core.kwargs["deterministic"] is False
    assert calc.core.kwargs["seed"] == 5
