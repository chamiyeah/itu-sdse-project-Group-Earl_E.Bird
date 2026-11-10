import inspect
import re

import mlflow
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vial_classifier.config import Config
from vial_classifier.models.cnn import SimpleCNN
from vial_classifier.seed import set_seed
from vial_classifier.training.threshold import tune_threshold

CPU = torch.device("cpu")
EXPECTED_GRID = [round(0.025 * step, 3) for step in range(1, 40)]


class FixedLogitModel(nn.Module):
    """Ignores the network: the logit of each image is stored in its first pixel."""

    def forward(self, images):
        return images[:, 0, 0, 0].view(-1, 1)


class CountingLoader:
    """Wraps a loader and counts how many times it is iterated."""

    def __init__(self, loader):
        self.loader = loader
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def make_loader(probabilities, labels):
    """Images of shape (3, 64, 64) that make FixedLogitModel return these probabilities."""
    images = torch.zeros(len(labels), 3, 64, 64)
    images[:, 0, 0, 0] = torch.logit(torch.tensor(probabilities))
    return DataLoader(TensorDataset(images, torch.tensor(labels)), batch_size=32, shuffle=False)


def make_two_group_loader(negative_probability, positive_probability):
    probabilities = [negative_probability] * 6 + [positive_probability] * 6
    labels = [0] * 6 + [1] * 6
    return make_loader(probabilities, labels)


def read_threshold_lines(output):
    """Return {threshold: macro-F1 text} from the lines printed for every threshold."""
    matches = re.findall(r"^  Threshold ([0-9.]+): MacroF1=([0-9.]+),", output, flags=re.MULTILINE)
    return {float(threshold): macro_f1 for threshold, macro_f1 in matches}


def read_active_run():
    return mlflow.get_run(mlflow.active_run().info.run_id)


def test_default_initial_threshold_matches_config():
    parameters = inspect.signature(tune_threshold).parameters

    assert parameters["initial_threshold"].default == Config().initial_threshold


def test_grid_has_39_values_from_0025_to_0975(mlflow_tracking, capsys):
    tune_threshold(FixedLogitModel(), make_two_group_loader(0.31, 0.69), CPU)

    grid = list(read_threshold_lines(capsys.readouterr().out))

    assert len(grid) == 39
    assert grid[0] == 0.025
    assert grid[-1] == 0.975
    assert grid == EXPECTED_GRID
    for previous, current in zip(grid, grid[1:]):
        assert current - previous == pytest.approx(0.025)


def test_golden_of_the_saved_run(mlflow_tracking, capsys):
    """All probabilities at 0.51 with the proportions of the validation set (22/86)."""
    probabilities = [0.51] * 108
    labels = [0] * 22 + [1] * 86

    with mlflow.start_run():
        result = tune_threshold(FixedLogitModel(), make_loader(probabilities, labels), CPU)
        logged_macro_f1 = read_active_run().data.metrics["optimal_threshold_macro_f1"]

    assert result == 0.025
    assert logged_macro_f1 == pytest.approx((172 / 194) / 2)

    macro_f1_by_threshold = read_threshold_lines(capsys.readouterr().out)
    assert len(macro_f1_by_threshold) == 39
    for threshold, macro_f1 in macro_f1_by_threshold.items():
        if threshold <= 0.5:
            assert macro_f1 == f"{(172 / 194) / 2:.4f}" == "0.4433"
        else:
            assert macro_f1 == f"{(44 / 130) / 2:.4f}" == "0.1692"


def test_ties_are_won_by_the_lowest_threshold_of_the_grid(mlflow_tracking):
    # Every threshold from 0.325 to 0.675 separates the groups: all have macro-F1 1.0
    result = tune_threshold(FixedLogitModel(), make_two_group_loader(0.31, 0.69), CPU)

    assert result == 0.325


def test_ties_are_won_by_the_first_threshold_of_the_range(mlflow_tracking):
    loader = make_two_group_loader(0.31, 0.69)

    assert tune_threshold(FixedLogitModel(), loader, CPU, threshold_range=[0.4, 0.6]) == 0.4


def test_macro_f1_always_zero_keeps_the_initial_threshold(mlflow_tracking, capsys):
    """Characterization test of nb01 c29: no threshold beats 0.0, so the start value stays.

    Negatives are above every threshold and positives below every threshold, so all the
    predictions are wrong on the whole grid.
    """
    loader = make_two_group_loader(0.99, 0.01)

    with mlflow.start_run():
        result = tune_threshold(FixedLogitModel(), loader, CPU, initial_threshold=0.123)
        run = read_active_run()

    assert result == 0.123
    assert type(result) is float
    assert run.data.params["optimal_threshold"] == "0.123"
    assert run.data.metrics["optimal_threshold_macro_f1"] == 0.0
    assert set(read_threshold_lines(capsys.readouterr().out).values()) == {"0.0000"}


def test_intermediate_threshold_wins_with_perfect_macro_f1(mlflow_tracking):
    loader = make_two_group_loader(0.3, 0.7)

    with mlflow.start_run():
        result = tune_threshold(FixedLogitModel(), loader, CPU)
        logged_macro_f1 = read_active_run().data.metrics["optimal_threshold_macro_f1"]

    assert 0.3 <= result <= 0.7
    assert logged_macro_f1 == 1.0


def test_probability_equal_to_threshold_counts_as_positive(mlflow_tracking):
    # torch.logit(0.5) is exactly 0 and torch.sigmoid(0) is exactly 0.5
    loader = make_two_group_loader(0.1, 0.5)

    with mlflow.start_run():
        tune_threshold(FixedLogitModel(), loader, CPU, threshold_range=[0.5])
        logged_macro_f1 = read_active_run().data.metrics["optimal_threshold_macro_f1"]

    assert logged_macro_f1 == 1.0


def test_returned_value_is_a_numpy_float_from_the_grid(mlflow_tracking):
    """Characterization test of nb01 c29: the grid values are np.float64, not float."""
    result = tune_threshold(FixedLogitModel(), make_two_group_loader(0.31, 0.69), CPU)

    assert type(result) is np.float64
    assert result in EXPECTED_GRID


def test_smoke_with_simple_cnn(mlflow_tracking):
    set_seed(42)
    images = torch.rand(16, 3, 64, 64)
    labels = torch.tensor([0] * 8 + [1] * 8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)

    result = tune_threshold(SimpleCNN(), loader, CPU)

    assert result in EXPECTED_GRID


def test_model_is_left_in_eval_mode_and_loader_is_iterated_once(mlflow_tracking):
    model = FixedLogitModel()
    model.train()
    loader = CountingLoader(make_two_group_loader(0.31, 0.69))

    tune_threshold(model, loader, CPU)

    assert model.training is False
    assert loader.iter_calls == 1


def test_logs_expected_names_inside_a_run(mlflow_tracking):
    with mlflow.start_run():
        tune_threshold(FixedLogitModel(), make_two_group_loader(0.31, 0.69), CPU)
        run = read_active_run()

    assert run.data.params == {"optimal_threshold": "0.325"}
    assert run.data.metrics == {"optimal_threshold_macro_f1": 1.0}


def test_without_active_run_mlflow_opens_one_and_leaves_it_open(mlflow_tracking):
    """Characterization test of nb01 c29: there is no mlflow.active_run() check.

    The first mlflow.log_param call starts a new run, and nothing closes it.
    """
    assert mlflow.active_run() is None

    tune_threshold(FixedLogitModel(), make_two_group_loader(0.31, 0.69), CPU)

    assert mlflow.active_run() is not None
    assert read_active_run().data.params == {"optimal_threshold": "0.325"}
