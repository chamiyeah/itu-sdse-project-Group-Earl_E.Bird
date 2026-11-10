import mlflow
import pytest

from vial_classifier.tracking import setup_mlflow_tracking


@pytest.fixture
def mlflow_paths(tmp_path):
    """Temporary MLflow paths, restoring the global tracking URI after the test."""
    state_dir = tmp_path / "mlflow-state"
    artifact_location = tmp_path / "mlflow-state" / "artifacts"
    # tmp_path starts with "/", so the URI has four slashes: sqlite:////tmp/...
    tracking_uri = f"sqlite:///{state_dir}/mlflow.db"

    original_uri = mlflow.get_tracking_uri()
    try:
        yield state_dir, tracking_uri, str(artifact_location)
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_creates_state_dir_and_sets_tracking_uri(mlflow_paths):
    state_dir, tracking_uri, artifact_location = mlflow_paths

    setup_mlflow_tracking(state_dir, tracking_uri, artifact_location)

    assert state_dir.is_dir()
    assert mlflow.get_tracking_uri() == tracking_uri


def test_creates_experiment_with_expected_name_and_artifact_location(mlflow_paths):
    state_dir, tracking_uri, artifact_location = mlflow_paths

    setup_mlflow_tracking(state_dir, tracking_uri, artifact_location, "vials")

    experiment = mlflow.get_experiment_by_name("vials")
    assert experiment is not None
    assert experiment.name == "vials"
    assert experiment.artifact_location == artifact_location


def test_second_call_does_not_create_another_experiment(mlflow_paths):
    state_dir, tracking_uri, artifact_location = mlflow_paths

    setup_mlflow_tracking(state_dir, tracking_uri, artifact_location, "vials")
    first_id = mlflow.get_experiment_by_name("vials").experiment_id
    count_after_first_call = len(mlflow.search_experiments())

    setup_mlflow_tracking(state_dir, tracking_uri, artifact_location, "vials")

    assert mlflow.get_experiment_by_name("vials").experiment_id == first_id
    assert len(mlflow.search_experiments()) == count_after_first_call


def test_without_experiment_name_no_experiment_is_created(mlflow_paths):
    state_dir, tracking_uri, artifact_location = mlflow_paths

    setup_mlflow_tracking(state_dir, tracking_uri, artifact_location)

    assert mlflow.get_experiment_by_name("vials") is None
