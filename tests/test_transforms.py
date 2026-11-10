import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from vial_classifier.config import Config
from vial_classifier.features.transforms import define_transforms
from vial_classifier.seed import set_seed


def make_transforms():
    config = Config()
    return define_transforms(config.image_size, config.transform_mean, config.transform_std)


def make_solid_image(color):
    return Image.new("RGB", (200, 96), color)


def make_varied_image():
    """Horizontal gradient that is different in every channel, so augmentation is visible."""
    pixels = np.zeros((96, 200, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.linspace(0, 255, 200).astype(np.uint8)
    pixels[:, :, 1] = np.linspace(255, 0, 200).astype(np.uint8)
    pixels[:, :, 2] = 80
    return Image.fromarray(pixels)


def test_config_tuples_are_accepted_by_resize_and_normalize():
    config = Config()

    train_transforms, val_transforms = make_transforms()
    result = val_transforms(make_varied_image())

    assert isinstance(config.image_size, tuple)
    assert isinstance(config.transform_mean, tuple)
    assert isinstance(config.transform_std, tuple)
    assert train_transforms.transforms[0].size == (64, 64)
    assert val_transforms.transforms[-1].mean == config.transform_mean
    assert val_transforms.transforms[-1].std == config.transform_std
    assert isinstance(result, torch.Tensor)


def test_train_and_val_output_shape():
    train_transforms, val_transforms = make_transforms()
    image = make_varied_image()

    assert train_transforms(image).shape == (3, 64, 64)
    assert val_transforms(image).shape == (3, 64, 64)


def test_val_is_deterministic():
    _, val_transforms = make_transforms()
    image = make_varied_image()

    assert torch.equal(val_transforms(image), val_transforms(image))


def test_val_normalizes_white_image():
    _, val_transforms = make_transforms()

    result = val_transforms(make_solid_image((255, 255, 255)))

    for channel, expected in enumerate([2.2489, 2.4286, 2.6400]):
        assert result[channel].min().item() == pytest.approx(expected, abs=1e-4)
        assert result[channel].max().item() == pytest.approx(expected, abs=1e-4)


def test_val_normalizes_black_image():
    config = Config()
    _, val_transforms = make_transforms()

    result = val_transforms(make_solid_image((0, 0, 0)))

    for channel in range(3):
        expected = -config.transform_mean[channel] / config.transform_std[channel]
        assert result[channel].min().item() == pytest.approx(expected, abs=1e-4)
        assert result[channel].max().item() == pytest.approx(expected, abs=1e-4)


def test_train_is_reproducible_with_same_seed():
    train_transforms, _ = make_transforms()
    image = make_varied_image()

    set_seed(42)
    first = train_transforms(image)
    set_seed(42)
    second = train_transforms(image)

    assert torch.equal(first, second)


def test_train_changes_with_different_seed():
    train_transforms, _ = make_transforms()
    image = make_varied_image()

    set_seed(42)
    first = train_transforms(image)
    set_seed(43)
    second = train_transforms(image)

    assert not torch.equal(first, second)


def test_transform_order_matches_notebook_cell_21():
    train_transforms, val_transforms = make_transforms()

    train_types = [type(step) for step in train_transforms.transforms]
    val_types = [type(step) for step in val_transforms.transforms]

    assert train_types == [
        transforms.Resize,
        transforms.RandomHorizontalFlip,
        transforms.RandomRotation,
        transforms.ColorJitter,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert val_types == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_augmentation_parameters_match_notebook_cell_21():
    train_transforms, _ = make_transforms()
    flip = train_transforms.transforms[1]
    rotation = train_transforms.transforms[2]
    jitter = train_transforms.transforms[3]

    assert flip.p == 0.5
    assert list(rotation.degrees) == [-10.0, 10.0]
    assert list(jitter.brightness) == [0.8, 1.2]
    assert list(jitter.contrast) == [0.8, 1.2]
    assert list(jitter.saturation) == [0.8, 1.2]
    assert jitter.hue is None
