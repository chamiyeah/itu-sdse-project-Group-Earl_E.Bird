from dataclasses import FrozenInstanceError

import pytest

from {{ cookiecutter.package_name }}.config import Config


def test_config_has_project_defaults():
    config = Config()

    assert config.experiment_name == "{{ cookiecutter.experiment_name }}"
    assert config.model_name == "{{ cookiecutter.model_name }}"
    assert config.mlflow_tracking_uri == "sqlite:///mlflow-state/mlflow.db"


def test_config_is_immutable():
    with pytest.raises(FrozenInstanceError):
        Config().num_epochs = 10
