import inspect

import mlflow
import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from vial_classifier.config import Config
from vial_classifier.models.cnn import SimpleCNN
from vial_classifier.seed import set_seed
from vial_classifier.training import train as train_module
from vial_classifier.training.train import train_cnn_model

CPU = torch.device("cpu")


class SyntheticDataset(Dataset):
    """Random images with a .labels list, the attribute train_cnn_model reads."""

    def __init__(self, num_negative, num_positive, data_seed):
        generator = torch.Generator().manual_seed(data_seed)
        total = num_negative + num_positive
        self.images = torch.rand(total, 3, 64, 64, generator=generator)
        self.labels = [0] * num_negative + [1] * num_positive

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CountingLoader:
    """Wraps a loader and counts how many times it is iterated."""

    def __init__(self, loader):
        self.loader = loader
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def make_train_loader(num_negative=18, num_positive=6):
    dataset = SyntheticDataset(num_negative, num_positive, data_seed=0)
    return DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)


def make_val_loader():
    dataset = SyntheticDataset(12, 4, data_seed=1)
    return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)


def run_training(**kwargs):
    return train_cnn_model(make_train_loader(), make_val_loader(), CPU, **kwargs)


def constant_f1(y_true, y_pred, average):
    return 0.5


def test_default_values_match_config():
    config = Config()
    parameters = inspect.signature(train_cnn_model).parameters

    assert parameters["num_epochs"].default == config.num_epochs
    assert parameters["learning_rate"].default == config.learning_rate
    assert parameters["random_seed"].default == config.random_state
    assert parameters["threshold"].default == config.initial_threshold
    assert parameters["early_stop_patience"].default == config.early_stop_patience
    assert parameters["scheduler_patience"].default == config.scheduler_patience


def test_same_seed_gives_same_weights_and_history():
    first_model, first_history = run_training(num_epochs=2, random_seed=7)
    second_model, second_history = run_training(num_epochs=2, random_seed=7)

    first_weights = first_model.state_dict()
    second_weights = second_model.state_dict()
    for key in first_weights:
        assert torch.equal(first_weights[key], second_weights[key])

    for key in first_history:
        if key == "best_model_state":
            continue
        assert first_history[key] == second_history[key]
    for key in first_history["best_model_state"]:
        assert torch.equal(first_history["best_model_state"][key], second_history["best_model_state"][key])


def test_different_seeds_give_different_weights():
    first_model, _ = run_training(num_epochs=1, random_seed=7)
    second_model, _ = run_training(num_epochs=1, random_seed=8)

    first_weights = first_model.state_dict()
    second_weights = second_model.state_dict()
    for key in first_weights:
        assert not torch.equal(first_weights[key], second_weights[key])


def test_model_is_created_right_after_the_seed(monkeypatch):
    states_at_creation = []

    def spy_model():
        states_at_creation.append(torch.get_rng_state().clone())
        return SimpleCNN()

    monkeypatch.setattr(train_module, "SimpleCNN", spy_model)
    set_seed(7)
    expected_state = torch.get_rng_state().clone()
    torch.rand(10)  # Move the generator away, so only the seed in train can bring it back

    run_training(num_epochs=1, random_seed=7)

    assert len(states_at_creation) == 1
    assert torch.equal(states_at_creation[0], expected_state)


def test_val_loader_is_iterated_once_per_epoch():
    val_loader = CountingLoader(make_val_loader())

    _, history = train_cnn_model(make_train_loader(), val_loader, CPU, num_epochs=3, early_stop_patience=100)

    assert len(history["val_loss"]) == 3
    assert val_loader.iter_calls == 3


def test_val_loader_is_iterated_once_per_epoch_with_early_stopping(monkeypatch):
    monkeypatch.setattr(train_module, "f1_score", constant_f1)
    val_loader = CountingLoader(make_val_loader())

    _, history = train_cnn_model(make_train_loader(), val_loader, CPU, num_epochs=5)

    assert len(history["val_loss"]) == 3
    assert val_loader.iter_calls == 3


def test_pos_weight_is_negatives_over_positives(monkeypatch):
    recorded_weights = []

    class SpyLoss(nn.BCEWithLogitsLoss):
        def __init__(self, pos_weight):
            recorded_weights.append(pos_weight.clone())
            super().__init__(pos_weight=pos_weight)

    monkeypatch.setattr(train_module.nn, "BCEWithLogitsLoss", SpyLoss)

    train_cnn_model(make_train_loader(num_negative=18, num_positive=6), make_val_loader(), CPU, num_epochs=1)

    assert len(recorded_weights) == 1
    assert recorded_weights[0].tolist() == [3.0]


def test_early_stopping_does_not_log_the_last_epoch(monkeypatch, mlflow_tracking, capsys):
    """Characterization test of a known bug in nb01 c27, not the desired behaviour.

    The break of the early stopping comes before the print and the log_metric calls, so
    the last epoch runs but is neither printed nor logged. Update this test when the bug
    is fixed.
    """
    monkeypatch.setattr(train_module, "f1_score", constant_f1)

    with mlflow.start_run() as run:
        _, history = run_training(num_epochs=5, early_stop_patience=2)

    assert len(history["train_loss"]) == 3

    client = mlflow.MlflowClient()
    for name in [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_f1_score",
        "val_f1_score",
        "val_f1_macro",
    ]:
        metric_steps = [metric.step for metric in client.get_metric_history(run.info.run_id, name)]
        assert sorted(metric_steps) == [0, 1]

    output = capsys.readouterr().out
    assert "Epoch [2/5]" in output
    assert "Epoch [3/5]" not in output
    assert "Early stopping" in output


def test_scheduler_never_lowers_learning_rate_before_early_stopping(monkeypatch):
    """Characterization test of a known oddity in nb01 c2/c27, not the desired behaviour.

    The scheduler patience (3) is larger than the early stopping patience (2), so training
    stops before the learning rate can drop. Update this test when this is fixed.
    """
    monkeypatch.setattr(train_module, "f1_score", constant_f1)
    learning_rates = []
    patiences = []

    class RecordingScheduler(ReduceLROnPlateau):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            patiences.append(self.patience)

        def step(self, metrics):
            super().step(metrics)
            learning_rates.append(self.optimizer.param_groups[0]["lr"])

    monkeypatch.setattr(train_module.optim.lr_scheduler, "ReduceLROnPlateau", RecordingScheduler)

    run_training(num_epochs=5)

    assert patiences == [3]
    assert learning_rates == [0.0003, 0.0003, 0.0003]


def test_returned_model_has_the_weights_of_the_best_epoch(monkeypatch):
    models = []
    weights_per_epoch = []
    macro_values = iter([0.2, 0.9, 0.1, 0.1])

    def spy_model():
        model = SimpleCNN()
        models.append(model)
        return model

    def fake_f1(y_true, y_pred, average):
        if average != "macro":
            return 0.5
        # Macro-F1 is computed after the validation pass: the weights of this epoch are final
        weights_per_epoch.append({key: value.clone() for key, value in models[0].state_dict().items()})
        return next(macro_values)

    monkeypatch.setattr(train_module, "SimpleCNN", spy_model)
    monkeypatch.setattr(train_module, "f1_score", fake_f1)

    model, history = run_training(num_epochs=4, early_stop_patience=2)

    assert history["best_val_f1_macro"] == 0.9
    assert len(weights_per_epoch) == 4
    final_weights = model.state_dict()
    for key in final_weights:
        assert torch.equal(final_weights[key], weights_per_epoch[1][key])
    assert any(not torch.equal(final_weights[key], weights_per_epoch[3][key]) for key in final_weights)


def test_no_positive_samples_raises_zero_division():
    """Characterization test of nb01 c27: num_negative / num_positive with zero positives."""
    with pytest.raises(ZeroDivisionError):
        train_cnn_model(make_train_loader(num_negative=24, num_positive=0), make_val_loader(), CPU, num_epochs=1)
