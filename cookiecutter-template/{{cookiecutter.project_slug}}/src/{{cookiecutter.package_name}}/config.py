from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration shared by training and inference."""

    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.0003
    random_state: int = 42
    artifact_dir: str = "artifacts"
    mlflow_state_dir: str = "mlflow-state"
    experiment_name: str = "{{ cookiecutter.experiment_name }}"
    model_name: str = "{{ cookiecutter.model_name }}"

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"sqlite:///{self.mlflow_state_dir}/mlflow.db"
