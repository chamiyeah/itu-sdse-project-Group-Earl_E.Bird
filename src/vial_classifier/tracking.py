import os

import mlflow


def setup_mlflow_tracking(state_dir, tracking_uri, artifact_location, experiment_name=None):
    """Point MLflow at the local state dir and create the experiment if needed."""
    os.makedirs(state_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name is None:
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
