import pytest
import torch
import torch.nn as nn

from vial_classifier.models.cnn import SimpleCNN, get_device
from vial_classifier.seed import set_seed


def test_forward_output_shape():
    model = SimpleCNN()
    model.eval()

    output = model(torch.zeros(4, 3, 64, 64))

    assert output.shape == (4, 1)


def test_number_of_parameters():
    model = SimpleCNN()

    assert sum(p.numel() for p in model.parameters()) == 2_116_801
    assert sum(p.numel() for p in model.conv1.parameters()) == 896
    assert sum(p.numel() for p in model.conv2.parameters()) == 18_496
    assert sum(p.numel() for p in model.fc1.parameters()) == 2_097_280
    assert sum(p.numel() for p in model.fc2.parameters()) == 129


def test_layer_settings_match_notebook_cell_26():
    model = SimpleCNN()

    for layer in (model.conv1, model.conv2):
        assert layer.kernel_size == (3, 3)
        assert layer.stride == (1, 1)
        assert layer.padding == (1, 1)
    assert (model.conv1.in_channels, model.conv1.out_channels) == (3, 32)
    assert (model.conv2.in_channels, model.conv2.out_channels) == (32, 64)
    assert model.pool.kernel_size == 2
    assert model.pool.stride == 2
    assert (model.fc1.in_features, model.fc1.out_features) == (16384, 128)
    assert (model.fc2.in_features, model.fc2.out_features) == (128, 1)


def test_module_order_matches_notebook_cell_26():
    model = SimpleCNN()

    names = [name for name, _ in model.named_children()]

    assert names == ["conv1", "conv2", "pool", "fc1", "fc2", "relu", "dropout"]


def test_state_dict_keys():
    model = SimpleCNN()

    assert list(model.state_dict().keys()) == [
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    ]


def test_same_seed_gives_same_initial_weights():
    set_seed(42)
    first = SimpleCNN().state_dict()
    set_seed(42)
    second = SimpleCNN().state_dict()

    for key in first:
        assert torch.equal(first[key], second[key])


def test_different_seeds_give_different_initial_weights():
    set_seed(42)
    first = SimpleCNN().state_dict()
    set_seed(43)
    second = SimpleCNN().state_dict()

    for key in first:
        assert not torch.equal(first[key], second[key])


def test_initial_weights_follow_the_layer_order_of_notebook_cell_26():
    # The same layers of c26, created by hand in the same order
    set_seed(42)
    conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    fc1 = nn.Linear(64 * 16 * 16, 128)
    fc2 = nn.Linear(128, 1)

    set_seed(42)
    model = SimpleCNN()

    assert torch.equal(model.conv1.weight, conv1.weight)
    assert torch.equal(model.conv2.weight, conv2.weight)
    assert torch.equal(model.fc1.weight, fc1.weight)
    assert torch.equal(model.fc2.weight, fc2.weight)


def test_dropout_probability():
    assert SimpleCNN().dropout.p == 0.3


def test_eval_mode_is_deterministic():
    model = SimpleCNN()
    model.eval()
    images = torch.rand(2, 3, 64, 64)

    assert torch.equal(model(images), model(images))


@pytest.mark.parametrize("cuda_available, expected", [(False, "cpu"), (True, "cuda")])
def test_get_device(monkeypatch, cuda_available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = get_device()

    assert device.type == expected
    assert device.type != "mps"
