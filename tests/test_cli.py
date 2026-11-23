import pytest

from vial_classifier import cli
from vial_classifier.config import Config, get_settings


@pytest.fixture
def calls(monkeypatch):
    """Replace the two real jobs with recorders: no network and no training."""
    recorded = {"acquire": [], "train": []}

    def fake_download_all_images(**kwargs):
        recorded["acquire"].append(kwargs)

    def fake_run_training(config):
        recorded["train"].append(config)

    monkeypatch.setattr(cli, "download_all_images", fake_download_all_images)
    monkeypatch.setattr(cli, "run_training", fake_run_training)
    return recorded


def test_train_calls_run_training_with_the_default_config(calls):
    exit_code = cli.main(["train"])

    assert exit_code == 0
    assert calls["train"] == [get_settings()]
    assert calls["acquire"] == []


def test_acquire_calls_download_all_images_with_the_config_values(calls):
    exit_code = cli.main(["acquire"])

    config = get_settings()
    assert exit_code == 0
    assert calls["train"] == []
    assert calls["acquire"] == [{
        "endpoint_url": config.s3_endpoint,
        "bucket": config.s3_bucket,
        "image_metadata_path": config.metadata_path,
        "image_data_dir": config.image_data_dir,
        "images_prefix": config.s3_images_prefix,
        "test_metadata_path": config.test_metadata_path,
        "test_image_dir": config.test_image_dir,
        "test_prefix": config.s3_test_prefix,
    }]


def test_unknown_command_returns_exit_code_2(calls, capsys):
    exit_code = cli.main(["deploy"])

    assert exit_code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert calls == {"acquire": [], "train": []}


def test_missing_command_returns_exit_code_2(calls):
    assert cli.main([]) == 2
    assert calls == {"acquire": [], "train": []}


def test_extra_arguments_are_rejected(calls):
    # No overrides from the command line in this unit.
    assert cli.main(["train", "--num-epochs", "1"]) == 2
    assert calls == {"acquire": [], "train": []}


def test_smoke_train_uses_one_epoch_and_synthetic_data(calls, monkeypatch):
    smoke_roots = []
    monkeypatch.setattr(cli, "create_smoke_dataset", smoke_roots.append)

    exit_code = cli.main(["train", "--mode", "smoke"])

    assert exit_code == 0
    assert len(smoke_roots) == 1
    assert calls["train"][0].num_epochs == 1
    assert calls["acquire"] == []
