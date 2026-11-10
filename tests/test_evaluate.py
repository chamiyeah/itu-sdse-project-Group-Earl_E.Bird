import inspect
import json

import mlflow
import pytest
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader, TensorDataset

from vial_classifier.config import Config
from vial_classifier.models.cnn import SimpleCNN
from vial_classifier.seed import set_seed
from vial_classifier.training.evaluate import evaluate_cnn_model

CPU = torch.device("cpu")


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


def make_degenerate_loader():
    """24 images of class 0 and 88 of class 1, the proportions of the test set, all at 0.51."""
    return make_loader([0.51] * 112, [0] * 24 + [1] * 88)


def read_active_run():
    return mlflow.get_run(mlflow.active_run().info.run_id)


def test_default_threshold_matches_config():
    parameters = inspect.signature(evaluate_cnn_model).parameters

    assert parameters["threshold"].default == Config().initial_threshold


def test_golden_degenerate_model(mlflow_tracking):
    result = evaluate_cnn_model(FixedLogitModel(), make_degenerate_loader(), CPU, threshold=0.5)

    assert result["confusion_matrix"] == [[0, 24], [0, 88]]
    assert result["accuracy"] == pytest.approx(88 / 112)
    assert result["precision"] == pytest.approx(88 / 112)
    assert result["recall"] == 1.0
    assert result["f1_score"] == pytest.approx(0.88)
    assert result["roc_auc"] == 0.5


def test_perfect_separation(mlflow_tracking):
    result = evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.1, 0.9), CPU, threshold=0.5)

    assert result["confusion_matrix"] == [[6, 0], [0, 6]]
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["roc_auc"] == 1.0


def test_inverted_separation(mlflow_tracking):
    result = evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.9, 0.1), CPU, threshold=0.5)

    assert result["confusion_matrix"] == [[0, 6], [6, 0]]
    assert result["accuracy"] == 0.0
    assert result["roc_auc"] == 0.0


def test_precision_and_recall_are_not_swapped(mlflow_tracking):
    # 2 false positives and 1 false negative: precision 3/5, recall 3/4
    probabilities = [0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    result = evaluate_cnn_model(FixedLogitModel(), make_loader(probabilities, labels), CPU, threshold=0.5)

    assert result["confusion_matrix"] == [[2, 2], [1, 3]]
    assert result["precision"] == pytest.approx(3 / 5)
    assert result["recall"] == pytest.approx(3 / 4)
    assert result["f1_score"] == pytest.approx(2 * 3 / (2 * 3 + 2 + 1))


def test_roc_auc_uses_probabilities_not_predictions(mlflow_tracking):
    # All above the threshold, so every prediction is positive, but the ranking is perfect
    result = evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.6, 0.8), CPU, threshold=0.5)

    assert result["confusion_matrix"] == [[0, 6], [0, 6]]
    assert result["roc_auc"] == 1.0


def test_threshold_decides_the_predictions(mlflow_tracking):
    loader = make_two_group_loader(0.6, 0.6)

    low = evaluate_cnn_model(FixedLogitModel(), loader, CPU, threshold=0.5)
    high = evaluate_cnn_model(FixedLogitModel(), loader, CPU, threshold=0.7)

    assert low["confusion_matrix"] == [[0, 6], [0, 6]]
    assert high["confusion_matrix"] == [[6, 0], [6, 0]]


def test_probability_equal_to_threshold_counts_as_positive(mlflow_tracking):
    # torch.logit(0.5) is exactly 0 and torch.sigmoid(0) is exactly 0.5
    result = evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.5, 0.5), CPU, threshold=0.5)

    assert result["confusion_matrix"] == [[0, 6], [0, 6]]


def test_result_keys_and_types(mlflow_tracking):
    """Characterization test of the dictionary returned by nb01 c28."""
    result = evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.1, 0.9), CPU, threshold=0.5)

    assert list(result) == [
        "accuracy",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    ]
    for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        assert isinstance(result[key], float)

    assert type(result["confusion_matrix"]) is list
    for row in result["confusion_matrix"]:
        assert type(row) is list
        for value in row:
            assert type(value) is int

    # Labels are floats in c28, so the classes are named "0.0" and "1.0"
    report = result["classification_report"]
    assert list(report) == ["0.0", "1.0", "accuracy", "macro avg", "weighted avg"]
    assert isinstance(report["accuracy"], float)
    assert list(report["1.0"]) == ["precision", "recall", "f1-score", "support"]


def test_result_can_be_saved_as_json(mlflow_tracking):
    """Characterization test: c37 saves this dictionary with json.dump."""
    result = evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.1, 0.9), CPU, threshold=0.5)

    assert json.loads(json.dumps(result)) == result


def test_logs_expected_names_and_values_inside_a_run(mlflow_tracking):
    loader = make_loader([0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1], [0, 0, 0, 0, 1, 1, 1, 1])

    with mlflow.start_run():
        result = evaluate_cnn_model(FixedLogitModel(), loader, CPU, threshold=0.5)
        run = read_active_run()

    assert run.data.params == {}
    assert run.data.metrics == pytest.approx({
        "test_accuracy": result["accuracy"],
        "test_f1_score": result["f1_score"],
        "test_precision": result["precision"],
        "test_recall": result["recall"],
        "test_roc_auc": result["roc_auc"],
    })


def test_without_active_run_mlflow_opens_one_and_leaves_it_open(mlflow_tracking):
    """Characterization test of nb01 c28: there is no mlflow.active_run() check.

    The first mlflow.log_metric call starts a new run, and nothing closes it.
    """
    assert mlflow.active_run() is None

    evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.1, 0.9), CPU, threshold=0.5)

    assert mlflow.active_run() is not None
    assert "test_f1_score" in read_active_run().data.metrics


def test_loader_is_iterated_once_and_model_is_left_in_eval_mode(mlflow_tracking):
    model = FixedLogitModel()
    model.train()
    loader = CountingLoader(make_two_group_loader(0.1, 0.9))

    evaluate_cnn_model(model, loader, CPU, threshold=0.5)

    assert model.training is False
    assert loader.iter_calls == 1


def test_smoke_with_simple_cnn(mlflow_tracking):
    set_seed(42)
    images = torch.rand(16, 3, 64, 64)
    labels = torch.tensor([0] * 8 + [1] * 8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)

    result = evaluate_cnn_model(SimpleCNN(), loader, CPU, threshold=0.5)

    for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc", "confusion_matrix", "classification_report"]:
        assert key in result


def test_prints_results_and_classification_report(mlflow_tracking, capsys):
    evaluate_cnn_model(FixedLogitModel(), make_two_group_loader(0.1, 0.9), CPU, threshold=0.5)

    output = capsys.readouterr().out
    assert "Threshold: 0.5" in output
    assert "F1-Score: 1.0000" in output
    assert "[FN, TP]] = [[6, 0], [0, 6]]" in output
    assert "weighted avg" in output


def test_degenerate_model_emits_sklearn_warnings(mlflow_tracking, recwarn):
    """Characterization test: class 0 is never predicted, and c28 does not hide the warnings."""
    evaluate_cnn_model(FixedLogitModel(), make_degenerate_loader(), CPU, threshold=0.5)

    undefined_metric_warnings = [w for w in recwarn.list if w.category is UndefinedMetricWarning]
    assert len(undefined_metric_warnings) == 6
    for warning in undefined_metric_warnings:
        assert "Precision is ill-defined" in str(warning.message)
