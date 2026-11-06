from dataclasses import FrozenInstanceError, replace

import pytest

from vial_classifier.config import Config


def test_defaults_match_notebook_cell_2():
    config = Config()

    assert config.num_epochs == 5
    assert config.batch_size == 32
    assert config.learning_rate == 0.0003
    assert config.initial_threshold == 0.5
    assert config.early_stop_patience == 2
    assert config.scheduler_patience == 3
    assert config.random_state == 42

    assert config.image_size == (64, 64)
    assert config.num_classes == 2
    assert config.transform_mean == (0.485, 0.456, 0.406)
    assert config.transform_std == (0.229, 0.224, 0.225)

    assert config.artifact_dir == "artifacts"
    assert config.mlflow_state_dir == "mlflow-state"
    assert config.image_data_dir == "./data/images/raw"
    assert config.metadata_path == "./data/images/metadata.csv"
    assert config.pytorch_model_path == "artifacts/vial_cnn_model.pth"
    assert config.model_results_path == "artifacts/model_results.json"
    assert config.drift_baseline_stats_path == "artifacts/drift_baseline_stats.json"
    assert config.class_names_path == "artifacts/class_names.json"
    assert config.image_statistics_path == "artifacts/image_statistics.json"
    assert config.inference_log_path == "artifacts/inference_log.json"

    assert config.experiment_name == "vial_image_classification"
    assert config.model_name == "cnn_model"
    assert config.mlflow_tracking_uri == "sqlite:///mlflow-state/mlflow.db"
    assert config.mlflow_artifact_location == "./mlflow-state/artifacts"


def test_s3_defaults_match_notebook_cells_10_and_11():
    config = Config()

    assert config.s3_endpoint == "https://itudsip.hel1.your-objectstorage.com"
    assert config.s3_bucket == "data"
    assert config.s3_images_prefix == "images/raw/"
    assert config.s3_test_prefix == "test/"


def test_test_set_paths_match_notebook_cell_11():
    config = Config()

    assert config.test_image_dir == "./data/test"
    assert config.test_metadata_path == "./data/test/metadata.csv"


@pytest.mark.parametrize(
    "removed_name",
    ["max_date", "min_date", "original_image_size", "test_data_dir", "threshold"],
)
def test_removed_settings_do_not_exist(removed_name):
    assert not hasattr(Config(), removed_name)


def test_config_is_immutable():
    config = Config()

    with pytest.raises(FrozenInstanceError):
        config.num_epochs = 10


def test_replace_returns_a_changed_copy():
    config = Config()

    changed = replace(config, num_epochs=1)

    assert changed.num_epochs == 1
    assert changed.batch_size == 32
    assert config.num_epochs == 5


def test_replace_updates_the_paths_built_from_a_field():
    changed = replace(Config(), artifact_dir="other", mlflow_state_dir="state")

    assert changed.pytorch_model_path == "other/vial_cnn_model.pth"
    assert changed.mlflow_tracking_uri == "sqlite:///state/mlflow.db"
