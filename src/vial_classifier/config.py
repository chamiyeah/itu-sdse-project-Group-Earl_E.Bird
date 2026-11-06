from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All the settings of the training job. Only data, nothing is read from disk."""

    # Training configuration
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.0003
    initial_threshold: float = 0.5
    early_stop_patience: int = 2
    scheduler_patience: int = 3
    random_state: int = 42

    # Image specifications
    image_size: tuple = (64, 64)
    num_classes: int = 2
    transform_mean: tuple = (0.485, 0.456, 0.406)
    transform_std: tuple = (0.229, 0.224, 0.225)

    # Paths (relative to the project root)
    artifact_dir: str = "artifacts"
    mlflow_state_dir: str = "mlflow-state"
    image_data_dir: str = "./data/images/raw"
    metadata_path: str = "./data/images/metadata.csv"

    # MLflow
    experiment_name: str = "vial_image_classification"
    model_name: str = "cnn_model"
    mlflow_artifact_location: str = "./mlflow-state/artifacts"

    # Object store with the images
    s3_endpoint: str = "https://itudsip.hel1.your-objectstorage.com"
    s3_bucket: str = "data"
    s3_images_prefix: str = "images/raw/"
    s3_test_prefix: str = "test/"

    # Paths built from the fields above, so they follow any change to them
    @property
    def pytorch_model_path(self):
        return self.artifact_dir + "/vial_cnn_model.pth"

    @property
    def model_results_path(self):
        return self.artifact_dir + "/model_results.json"

    @property
    def drift_baseline_stats_path(self):
        return self.artifact_dir + "/drift_baseline_stats.json"

    @property
    def class_names_path(self):
        return self.artifact_dir + "/class_names.json"

    @property
    def image_statistics_path(self):
        return self.artifact_dir + "/image_statistics.json"

    @property
    def inference_log_path(self):
        return self.artifact_dir + "/inference_log.json"

    @property
    def mlflow_tracking_uri(self):
        return f"sqlite:///{self.mlflow_state_dir}/mlflow.db"
