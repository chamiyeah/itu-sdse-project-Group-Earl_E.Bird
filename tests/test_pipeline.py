import json
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from synthetic_data import make_synthetic_dataset
from vial_classifier import pipeline
from vial_classifier.config import Config
from vial_classifier.models.cnn import SimpleCNN
from vial_classifier.pipeline import run_training

# These tests do NOT use the mlflow_tracking fixture: run_training sets up MLflow
# by itself from Config. The autouse fixture in conftest.py has already moved us
# into tmp_path, so the relative paths in Config (artifacts/, data/) land in the
# temporary folder, and it closes any run left open at the end.


def make_config(tmp_path, **changes):
    """Config with 1 epoch and the MLflow state in an absolute path inside tmp_path.

    MLflow keeps one database connection per tracking URI string. With the default
    relative URI (sqlite:///mlflow-state/mlflow.db) every test would have the same
    string, so from the second test on the runs would end up in the database of the
    first test. An absolute path gives every test its own URI, like conftest.py does.
    """
    state_dir = tmp_path / "mlflow-state"
    return replace(
        Config(),
        num_epochs=1,
        mlflow_state_dir=str(state_dir),
        mlflow_artifact_location=str(state_dir / "artifacts"),
        **changes,
    )


def test_run_training_smoke(tmp_path):
    make_synthetic_dataset(tmp_path)
    result = run_training(make_config(tmp_path))

    # The four artifacts exist, at the paths that run_training returns
    expected_files = {
        "class_names": "artifacts/class_names.json",
        "drift_baseline_stats": "artifacts/drift_baseline_stats.json",
        "pytorch_model": "artifacts/vial_cnn_model.pth",
        "model_results": "artifacts/model_results.json",
    }
    assert result["artifact_paths"] == expected_files
    for path in expected_files.values():
        assert (tmp_path / path).is_file()
    # image_statistics.json is no longer written: nobody read it
    assert not (tmp_path / "artifacts" / "image_statistics.json").exists()

    # The JSON files are valid
    with open("artifacts/class_names.json") as f:
        assert json.load(f) == {"0": "good", "1": "defective"}

    with open("artifacts/drift_baseline_stats.json") as f:
        baseline = json.load(f)
    assert baseline["class_distribution"] == {"class_0_count": 3, "class_1_count": 6, "total": 9}

    with open("artifacts/model_results.json") as f:
        saved_results = json.load(f)
    assert saved_results == result["test_results"]

    # The checkpoint has the keys of nb01 c37
    checkpoint = torch.load("artifacts/vial_cnn_model.pth", weights_only=False)
    assert set(checkpoint) == {
        "model_state_dict", "architecture", "input_shape", "output_shape", "threshold", "num_classes",
    }
    assert checkpoint["architecture"] == "SimpleCNN"
    assert checkpoint["input_shape"] == (3, 64, 64)
    assert checkpoint["output_shape"] == (1,)
    assert checkpoint["num_classes"] == 2
    # Same value, but stored as a plain float instead of a np.float64
    assert checkpoint["threshold"] == result["optimal_threshold"]
    assert type(checkpoint["threshold"]) is float
    assert set(checkpoint["model_state_dict"]) == {
        "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
    }

    # One MLflow run, closed, with the threshold parameter
    assert mlflow.active_run() is None
    runs = mlflow.search_runs(experiment_names=["vial_image_classification"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "training"
    assert run.data.params["optimal_threshold"] == str(result["optimal_threshold"])
    assert "classification_threshold" not in run.data.params
    assert run.data.params["num_epochs"] == "1"
    assert run.data.params["image_size"] == "64x64"

    # pipeline.py does not create the MLflow artifact folder: MLflow creates it
    # when log_baseline_to_mlflow uploads the baseline file
    logged = list((tmp_path / "mlflow-state" / "artifacts").rglob("drift_detection/drift_baseline_stats.json"))
    assert len(logged) == 1


def test_run_training_reads_the_data_paths_from_config(tmp_path):
    make_synthetic_dataset(tmp_path)
    Path("data").rename("other_data")
    config = make_config(
        tmp_path,
        image_data_dir="other_data/images/raw",
        metadata_path="other_data/images/metadata.csv",
        artifact_dir="out",
    )

    result = run_training(config)

    assert result["artifact_paths"]["pytorch_model"] == "out/vial_cnn_model.pth"
    assert (tmp_path / "out" / "vial_cnn_model.pth").is_file()
    assert not (tmp_path / "artifacts").exists()


def test_run_is_closed_and_failed_when_a_step_fails(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path)

    def broken_tune_threshold(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "tune_threshold", broken_tune_threshold)

    with pytest.raises(RuntimeError, match="boom"):
        run_training(make_config(tmp_path))

    # The `with mlflow.start_run()` block closed the run and marked it as failed
    assert mlflow.active_run() is None
    runs = mlflow.search_runs(experiment_names=["vial_image_classification"], output_format="list")
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"


def test_optimal_threshold_is_passed_on_without_conversion(tmp_path, monkeypatch):
    """The probabilities are float32. With NumPy 2, comparing them to a np.float64
    happens in float64 (like in tune_threshold), but comparing them to a Python
    float happens in float32: a float() here could flip a prediction that sits
    right on the threshold, so evaluation and baseline must get the np.float64."""
    make_synthetic_dataset(tmp_path)
    received = {}
    real_evaluate = pipeline.evaluate_cnn_model
    real_baseline = pipeline.compute_baseline_statistics

    def recording_evaluate(model, loader, device, threshold):
        received["evaluate"] = threshold
        return real_evaluate(model, loader, device, threshold=threshold)

    def recording_baseline(model, loader, device, threshold):
        received["baseline"] = threshold
        return real_baseline(model, loader, device, threshold=threshold)

    monkeypatch.setattr(pipeline, "evaluate_cnn_model", recording_evaluate)
    monkeypatch.setattr(pipeline, "compute_baseline_statistics", recording_baseline)

    result = run_training(make_config(tmp_path))

    # The very same object goes to both steps and into the result
    assert received["evaluate"] is result["optimal_threshold"]
    assert received["baseline"] is result["optimal_threshold"]
    assert isinstance(result["optimal_threshold"], np.float64)


def test_logged_parameter_counts_match_the_trained_model(tmp_path):
    make_synthetic_dataset(tmp_path)

    run_training(make_config(tmp_path))

    checkpoint = torch.load("artifacts/vial_cnn_model.pth", weights_only=False)
    model = SimpleCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    expected = sum(p.numel() for p in model.parameters())

    run = mlflow.search_runs(experiment_names=["vial_image_classification"], output_format="list")[0]
    assert run.data.params["total_parameters"] == str(expected)
    assert run.data.params["trainable_parameters"] == str(expected)
