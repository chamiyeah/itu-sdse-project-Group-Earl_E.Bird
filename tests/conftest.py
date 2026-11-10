import os

import mlflow
import pytest

from vial_classifier.tracking import setup_mlflow_tracking


@pytest.fixture(autouse=True)
def isolate_test_state(tmp_path, monkeypatch):
    """Run every test in its own folder, without MLflow state from other tests."""
    monkeypatch.chdir(tmp_path)
    original_uri = mlflow.get_tracking_uri()
    original_experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    try:
        yield
    finally:
        while mlflow.active_run():
            mlflow.end_run()
        mlflow.set_tracking_uri(original_uri)
        if original_experiment_id is None:
            os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        else:
            os.environ["MLFLOW_EXPERIMENT_ID"] = original_experiment_id


@pytest.fixture
def mlflow_tracking(tmp_path):
    """Point MLflow at a temporary SQLite db with an experiment ready to use."""
    state_dir = tmp_path / "mlflow-state"
    # tmp_path starts with "/", so the URI has four slashes: sqlite:////tmp/...
    tracking_uri = f"sqlite:///{state_dir}/mlflow.db"
    artifact_location = str(state_dir / "artifacts")
    setup_mlflow_tracking(state_dir, tracking_uri, artifact_location, "test_experiment")
