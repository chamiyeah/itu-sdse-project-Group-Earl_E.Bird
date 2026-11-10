import inspect
import json

import mlflow
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vial_classifier.config import Config
from vial_classifier.drift.baseline import compute_baseline_statistics, log_baseline_to_mlflow
from vial_classifier.models.cnn import SimpleCNN
from vial_classifier.seed import set_seed

CPU = torch.device("cpu")


class FixedProbabilityModel(nn.Module):
    """Returns the same probability for every image, with a layer called fc1 like SimpleCNN."""

    def __init__(self, probability):
        super().__init__()
        self.fc1 = nn.Identity()
        self.logit = torch.logit(torch.tensor(probability))

    def forward(self, images):
        self.fc1(images[:, :, 0, 0])
        return torch.full((images.shape[0], 1), self.logit.item())


class FailingModel(FixedProbabilityModel):
    """Goes through fc1 and then fails, to check what happens to the hook."""

    def forward(self, images):
        self.fc1(images[:, :, 0, 0])
        raise RuntimeError("forward failed")


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


def make_loader(num_class_0, num_class_1):
    """Black images of shape (3, 64, 64) with the given number of images per class."""
    total = num_class_0 + num_class_1
    images = torch.zeros(total, 3, 64, 64)
    labels = torch.tensor([0] * num_class_0 + [1] * num_class_1)
    return DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=False)


def make_random_loader():
    images = torch.rand(16, 3, 64, 64)
    labels = torch.tensor([0] * 8 + [1] * 8)
    return DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)


def compute_with_simple_cnn():
    set_seed(42)
    model = SimpleCNN()
    return compute_baseline_statistics(model, make_random_loader(), CPU, threshold=0.5)


def make_baseline_stats():
    return {
        'image_statistics': {
            'mean_r': 0.1, 'mean_g': 0.2, 'mean_b': 0.3,
            'std_r': 0.4, 'std_g': 0.5, 'std_b': 0.6,
        },
        'class_distribution': {'class_0_count': 22, 'class_1_count': 86, 'total': 108},
        'prediction_distribution': {'pred_0_count': 0, 'pred_1_count': 108, 'total': 108},
        'confidence_statistics': {'mean': 0.7, 'std': 0.8, 'min': 0.0, 'max': 1.0},
        'embedding_statistics': {'mean': [0.0], 'std': [1.0], 'shape': 1},
    }


def test_default_values():
    parameters = inspect.signature(compute_baseline_statistics).parameters

    assert parameters["feature_layer"].default == "fc1"
    assert parameters["threshold"].default == Config().initial_threshold


def test_dictionary_structure_and_types():
    stats = compute_with_simple_cnn()

    assert list(stats) == [
        "image_statistics",
        "class_distribution",
        "prediction_distribution",
        "confidence_statistics",
        "embedding_statistics",
    ]
    assert list(stats["image_statistics"]) == ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
    assert list(stats["class_distribution"]) == ["class_0_count", "class_1_count", "total"]
    assert list(stats["prediction_distribution"]) == ["pred_0_count", "pred_1_count", "total"]
    assert list(stats["confidence_statistics"]) == ["mean", "std", "min", "max"]
    assert list(stats["embedding_statistics"]) == ["mean", "std", "shape"]

    for group in ["image_statistics", "confidence_statistics"]:
        for value in stats[group].values():
            assert type(value) is float
    for group in ["class_distribution", "prediction_distribution"]:
        for value in stats[group].values():
            assert type(value) is int


def test_embedding_statistics_have_128_values_with_simple_cnn():
    embedding = compute_with_simple_cnn()["embedding_statistics"]

    assert embedding["shape"] == 128
    assert type(embedding["shape"]) is int
    assert len(embedding["mean"]) == 128
    assert len(embedding["std"]) == 128
    assert type(embedding["mean"][0]) is float
    assert type(embedding["std"][0]) is float


def test_embedding_is_the_raw_output_of_fc1():
    """Characterization test of nb01 c33: the hook sees fc1 before ReLU and dropout."""
    set_seed(42)
    model = SimpleCNN()
    model.eval()
    loader = make_random_loader()
    images = torch.cat([batch_images for batch_images, _ in loader])
    with torch.no_grad():
        features = model.pool(model.relu(model.conv1(images)))
        features = model.pool(model.relu(model.conv2(features)))
        raw_output = model.fc1(features.view(-1, 64 * 16 * 16)).numpy()

    stats = compute_baseline_statistics(model, loader, CPU, threshold=0.5)

    # c33 runs batches of 8 and this test all 16 images: float32 sums differ around 1e-7
    assert stats["embedding_statistics"]["mean"] == pytest.approx(raw_output.mean(axis=0).tolist(), abs=1e-6)
    assert stats["embedding_statistics"]["std"] == pytest.approx(raw_output.std(axis=0).tolist(), abs=1e-6)
    assert min(stats["embedding_statistics"]["mean"]) < 0


def test_feature_layer_parameter_chooses_the_layer():
    set_seed(42)
    stats = compute_baseline_statistics(SimpleCNN(), make_random_loader(), CPU, feature_layer="fc2")

    assert stats["embedding_statistics"]["shape"] == 1


def test_class_distribution_counts_true_labels():
    stats = compute_baseline_statistics(FixedProbabilityModel(0.51), make_loader(22, 86), CPU, threshold=0.025)

    assert stats["class_distribution"] == {"class_0_count": 22, "class_1_count": 86, "total": 108}


def test_prediction_distribution_uses_the_threshold():
    loader = make_loader(22, 86)

    low = compute_baseline_statistics(FixedProbabilityModel(0.51), loader, CPU, threshold=0.025)
    high = compute_baseline_statistics(FixedProbabilityModel(0.51), loader, CPU, threshold=0.9)

    assert low["prediction_distribution"] == {"pred_0_count": 0, "pred_1_count": 108, "total": 108}
    assert high["prediction_distribution"] == {"pred_0_count": 108, "pred_1_count": 0, "total": 108}


def test_probability_equal_to_threshold_counts_as_positive():
    # torch.logit(0.5) is exactly 0 and torch.sigmoid(0) is exactly 0.5
    stats = compute_baseline_statistics(FixedProbabilityModel(0.5), make_loader(1, 1), CPU, threshold=0.5)

    assert stats["prediction_distribution"] == {"pred_0_count": 0, "pred_1_count": 2, "total": 2}


def test_confidence_statistics_with_constant_probability():
    stats = compute_baseline_statistics(FixedProbabilityModel(0.51), make_loader(22, 86), CPU, threshold=0.025)

    confidence = stats["confidence_statistics"]
    assert confidence["mean"] == pytest.approx(0.51, abs=1e-5)
    assert confidence["std"] == pytest.approx(0.0, abs=1e-5)
    assert confidence["min"] == pytest.approx(0.51, abs=1e-5)
    assert confidence["max"] == pytest.approx(0.51, abs=1e-5)


def test_image_statistics_keep_the_channels_apart():
    images = torch.zeros(4, 3, 64, 64)
    images[:, 0] = 0.1
    images[:, 1] = 0.2
    images[:, 2] = 0.3
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 0, 1, 1])), batch_size=4)

    stats = compute_baseline_statistics(FixedProbabilityModel(0.51), loader, CPU)["image_statistics"]

    assert stats["mean_r"] == pytest.approx(0.1)
    assert stats["mean_g"] == pytest.approx(0.2)
    assert stats["mean_b"] == pytest.approx(0.3)


def test_image_statistics_are_means_of_batch_values():
    """Characterization test of nb01 c33: every batch weighs the same, whatever its size.

    With 108 images and batch 32 the batches have 32, 32, 32 and 12 images. The first 96
    images are black (0.0) and the last 12 are white (1.0), so the last batch has mean 1.0.
    """
    images = torch.zeros(108, 3, 64, 64)
    images[96:] = 1.0
    labels = torch.tensor([0] * 22 + [1] * 86)
    loader = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=False)

    stats = compute_baseline_statistics(FixedProbabilityModel(0.51), loader, CPU)["image_statistics"]

    # Mean: mean of the 4 batch means, not the mean over all the images
    assert stats["mean_r"] == pytest.approx((0.0 + 0.0 + 0.0 + 1.0) / 4)
    assert images[:, 0].mean().item() == pytest.approx(12 / 108)
    assert stats["mean_r"] != pytest.approx(12 / 108)

    # Std: every batch is uniform, so the batch std is 0, while the std over all the images is not
    assert stats["std_r"] == 0.0
    assert images[:, 0].std().item() > 0.3


def test_hook_is_removed_after_a_normal_call():
    model = FixedProbabilityModel(0.51)

    compute_baseline_statistics(model, make_loader(1, 1), CPU)

    assert len(model.fc1._forward_hooks) == 0


def test_missing_layer_raises_value_error():
    with pytest.raises(ValueError, match="Model does not have layer 'missing'"):
        compute_baseline_statistics(FixedProbabilityModel(0.51), make_loader(1, 1), CPU, feature_layer="missing")


def test_hook_is_removed_after_an_error_in_forward():
    """Characterization test of nb01 c33: try/finally removes the hook and the error goes on."""
    model = FailingModel(0.51)

    with pytest.raises(RuntimeError, match="forward failed"):
        compute_baseline_statistics(model, make_loader(1, 1), CPU)

    assert len(model.fc1._forward_hooks) == 0


def test_loader_is_iterated_once_and_model_is_left_in_eval_mode():
    model = FixedProbabilityModel(0.51)
    model.train()
    loader = CountingLoader(make_loader(1, 1))

    compute_baseline_statistics(model, loader, CPU)

    assert model.training is False
    assert loader.iter_calls == 1


def test_dictionary_can_be_saved_as_json():
    """Characterization test: c33 saves this dictionary with json.dump."""
    stats = compute_with_simple_cnn()

    assert json.loads(json.dumps(stats)) == stats


def test_compute_does_not_use_mlflow_and_does_not_write_files(tmp_path):
    compute_with_simple_cnn()

    assert mlflow.active_run() is None
    assert list(tmp_path.iterdir()) == []


def test_log_baseline_logs_the_metrics_and_the_artifact_of_cell_33(mlflow_tracking, tmp_path):
    baseline_stats_path = tmp_path / "drift_baseline_stats.json"
    baseline_stats_path.write_text("{}")

    with mlflow.start_run() as run:
        log_baseline_to_mlflow(make_baseline_stats(), baseline_stats_path)

    finished_run = mlflow.get_run(run.info.run_id)
    assert finished_run.data.params == {}
    assert finished_run.data.metrics == {
        "baseline_image_mean_r": 0.1,
        "baseline_image_mean_g": 0.2,
        "baseline_image_mean_b": 0.3,
        "baseline_image_std_r": 0.4,
        "baseline_image_std_g": 0.5,
        "baseline_image_std_b": 0.6,
        "baseline_class_0_count": 22,
        "baseline_class_1_count": 86,
        "baseline_confidence_mean": 0.7,
        "baseline_confidence_std": 0.8,
    }
    artifacts = mlflow.MlflowClient().list_artifacts(run.info.run_id, "drift_detection")
    assert [artifact.path for artifact in artifacts] == ["drift_detection/drift_baseline_stats.json"]


def test_log_baseline_without_active_run_opens_one_and_leaves_it_open(mlflow_tracking, tmp_path):
    """Characterization test of nb01 c33: there is no mlflow.active_run() check."""
    baseline_stats_path = tmp_path / "drift_baseline_stats.json"
    baseline_stats_path.write_text("{}")
    assert mlflow.active_run() is None

    log_baseline_to_mlflow(make_baseline_stats(), baseline_stats_path)

    assert mlflow.active_run() is not None
    run = mlflow.get_run(mlflow.active_run().info.run_id)
    assert run.data.metrics["baseline_class_1_count"] == 86


def test_log_baseline_with_missing_file_fails_after_logging_the_metrics(mlflow_tracking, tmp_path):
    """Characterization test: c33 writes the file first, so pipeline.py must do the same."""
    with mlflow.start_run() as run:
        with pytest.raises(FileNotFoundError):
            log_baseline_to_mlflow(make_baseline_stats(), tmp_path / "missing.json")

    assert mlflow.get_run(run.info.run_id).data.metrics["baseline_class_1_count"] == 86
