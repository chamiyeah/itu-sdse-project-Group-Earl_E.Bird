import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchvision import transforms

from vial_classifier.data.dataset import (
    GlassVialDataset,
    calculate_image_statistics,
    create_dataloaders,
    load_image_dataset,
    split_dataset,
)


def make_fake_samples():
    """153 samples of class 0 and 577 of class 1, as in the real dataset. No file exists."""
    clean = [(Path(f"fake/clean_{i}.jpg"), 0) for i in range(153)]
    defective = [(Path(f"fake/defective_{i}.jpg"), 1) for i in range(577)]
    return clean + defective


def count_labels(samples):
    labels = [label for _, label in samples]
    return labels.count(0), labels.count(1)


def test_split_golden_counts():
    train, val, test = split_dataset(make_fake_samples())

    assert len(train) == 510
    assert len(val) == 108
    assert len(test) == 112
    assert count_labels(train) == (107, 403)
    assert count_labels(val) == (22, 86)
    assert count_labels(test) == (24, 88)


def test_split_has_no_overlap_and_keeps_all_samples():
    samples = make_fake_samples()

    train, val, test = split_dataset(samples)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(samples)


def test_same_seed_gives_same_split_in_same_order():
    samples = make_fake_samples()

    assert split_dataset(samples, random_seed=7) == split_dataset(samples, random_seed=7)


def test_different_seeds_give_different_split():
    samples = make_fake_samples()

    assert split_dataset(samples, random_seed=7) != split_dataset(samples, random_seed=8)


def test_default_seed_is_42():
    samples = make_fake_samples()

    assert split_dataset(samples) == split_dataset(samples, random_seed=42)


def test_split_uses_the_five_shuffles_of_notebook_cell_23():
    samples = make_fake_samples()

    # The same steps of c23, written out by hand
    np.random.seed(42)
    class_0 = samples[:153]
    class_1 = samples[153:]
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)
    expected_train = class_0[:107] + class_1[:403]
    expected_val = class_0[107:129] + class_1[403:489]
    expected_test = class_0[129:] + class_1[489:]
    np.random.shuffle(expected_train)
    np.random.shuffle(expected_val)
    np.random.shuffle(expected_test)

    train, val, test = split_dataset(samples)

    assert train == expected_train
    assert val == expected_val
    assert test == expected_test


def test_split_seeds_torch_without_consuming_it():
    torch.manual_seed(42)
    expected_state = torch.get_rng_state()
    torch.manual_seed(0)

    split_dataset(make_fake_samples())

    assert torch.equal(torch.get_rng_state(), expected_state)


def test_dataloader_batch_counts():
    train, val, test = split_dataset(make_fake_samples())

    train_loader, val_loader, test_loader = create_dataloaders(train, val, test, None, None, 32)

    assert len(train_loader) == 16
    assert len(val_loader) == 4
    assert len(test_loader) == 4


def test_dataloader_settings_match_notebook_cell_24():
    train, val, test = split_dataset(make_fake_samples())

    train_loader, val_loader, test_loader = create_dataloaders(train, val, test, None, None, 32)

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    for loader in (train_loader, val_loader, test_loader):
        assert loader.num_workers == 0
        assert loader.pin_memory is True
        assert loader.generator is None


def test_dataset_returns_three_channels_for_rgb_and_grayscale(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    gray_path = tmp_path / "gray.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(rgb_path)
    Image.new("L", (8, 6), 100).save(gray_path)

    dataset = GlassVialDataset([rgb_path, gray_path], [0, 1], transforms.ToTensor())

    assert len(dataset) == 2
    rgb_image, rgb_label = dataset[0]
    gray_image, gray_label = dataset[1]
    assert rgb_image.shape == (3, 6, 8)
    assert gray_image.shape == (3, 6, 8)
    assert rgb_label == 0
    assert gray_label == 1


def test_load_image_dataset_skips_missing_file(tmp_path, capsys):
    (tmp_path / "a.jpg").write_bytes(b"not read")
    (tmp_path / "c.jpg").write_bytes(b"not read")
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("filename,class\na.jpg,0\nb.jpg,1\nc.jpg,1\n")

    samples = load_image_dataset(tmp_path, metadata_path)

    assert samples == [(tmp_path / "a.jpg", 0), (tmp_path / "c.jpg", 1)]
    assert "Warning: 1 image files not found" in capsys.readouterr().out


def test_load_image_dataset_without_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_dataset(tmp_path, tmp_path / "missing.csv")


def test_image_statistics_keys_and_values():
    # 5 images, 3 channels, 4x4 pixels. Every channel has one constant value.
    images = torch.zeros(5, 3, 4, 4)
    images[:, 0] = 0.25
    images[:, 1] = 0.5
    images[:, 2] = 0.75
    labels = torch.zeros(5)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    stats = calculate_image_statistics(loader)

    assert set(stats) == {"mean", "std", "total_samples"}
    assert stats["mean"] == pytest.approx([0.25, 0.5, 0.75])
    assert stats["std"] == pytest.approx([0.0, 0.0, 0.0])
    assert stats["total_samples"] == 5
    json.dumps(stats)
