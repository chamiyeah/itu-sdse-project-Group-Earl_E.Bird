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
from vial_classifier.pipeline import run_training

# These tests do NOT use the mlflow_tracking fixture: run_training sets up MLflow
# by itself from Config. The autouse fixture in conftest.py has already moved us
# into tmp_path, so the relative paths in Config (artifacts/, mlflow-state/, data/)
# all land in the temporary folder, and it closes any run left open at the end.


def test_run_training_smoke(tmp_path):
    make_synthetic_dataset(tmp_path)
    config = replace(Config(), num_epochs=1)

    result = run_training(config)

    # The five artifacts exist, at the paths that run_training returns
    expected_files = {
        "image_statistics": "artifacts/image_statistics.json",
        "class_names": "artifacts/class_names.json",
        "drift_baseline_stats": "artifacts/drift_baseline_stats.json",
        "pytorch_model": "artifacts/vial_cnn_model.pth",
        "model_results": "artifacts/model_results.json",
    }
    assert result["artifact_paths"] == expected_files
    for path in expected_files.values():
        assert (tmp_path / path).is_file()

    # The JSON files are valid
    with open("artifacts/image_statistics.json") as f:
        image_stats = json.load(f)
    assert set(image_stats) == {"mean", "std", "total_samples"}
    assert image_stats["total_samples"] == 42

    with open("artifacts/class_names.json") as f:
        assert json.load(f) == {"0": "good", "1": "defective"}

    with open("artifacts/drift_baseline_stats.json") as f:
        baseline = json.load(f)
    assert baseline["class_distribution"] == {"class_0_count": 3, "class_1_count": 6, "total": 9}

    with open("artifacts/model_results.json") as f:
        saved_results = json.load(f)
    assert saved_results == result["test_results"]

    # The checkpoint has the keys of cell 37
    checkpoint = torch.load("artifacts/vial_cnn_model.pth", weights_only=False)
    assert set(checkpoint) == {
        "model_state_dict", "architecture", "input_shape", "output_shape", "threshold", "num_classes",
    }
    assert checkpoint["architecture"] == "SimpleCNN"
    assert checkpoint["input_shape"] == (3, 64, 64)
    assert checkpoint["output_shape"] == (1,)
    assert checkpoint["num_classes"] == 2
    assert checkpoint["threshold"] == result["optimal_threshold"]
    assert type(checkpoint["threshold"]) is type(result["optimal_threshold"])
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
    assert run.info.run_name == "training_notebook"
    assert run.data.params["classification_threshold"] == str(result["optimal_threshold"])
    assert run.data.params["num_epochs"] == "1"


def test_run_training_reads_the_data_paths_from_config(tmp_path):
    make_synthetic_dataset(tmp_path)
    Path("data").rename("other_data")
    config = replace(
        Config(),
        num_epochs=1,
        image_data_dir="other_data/images/raw",
        metadata_path="other_data/images/metadata.csv",
        artifact_dir="out",
    )

    result = run_training(config)

    assert result["artifact_paths"]["pytorch_model"] == "out/vial_cnn_model.pth"
    assert (tmp_path / "out" / "vial_cnn_model.pth").is_file()
    assert not (tmp_path / "artifacts").exists()


def test_run_stays_open_when_a_step_fails(tmp_path, monkeypatch):
    """Oddity copied from the notebook: there is no try/finally around the run."""
    make_synthetic_dataset(tmp_path)

    def broken_tune_threshold(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "tune_threshold", broken_tune_threshold)

    with pytest.raises(RuntimeError, match="boom"):
        run_training(replace(Config(), num_epochs=1))

    assert mlflow.active_run() is not None


def test_optimal_threshold_is_passed_on_without_conversion(tmp_path, monkeypatch):
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

    result = run_training(replace(Config(), num_epochs=1))

    # The very same object goes to both steps and into the result
    assert received["evaluate"] is result["optimal_threshold"]
    assert received["baseline"] is result["optimal_threshold"]
    assert isinstance(result["optimal_threshold"], np.float64)
