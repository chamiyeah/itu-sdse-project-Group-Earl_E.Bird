from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vial_classifier.seed import set_seed


def load_image_dataset(image_data_dir, metadata_path):
    """Load (image_path, label) pairs from the raw images and the metadata."""
    print(f"Loading image dataset from {image_data_dir}")
    print(f"Using metadata from {metadata_path}")

    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"Metadata loaded. Total entries: {len(metadata)}")

    samples = []
    missing_files = []
    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = Path(image_data_dir) / filename
        if image_path.exists():
            samples.append((image_path, label))
        else:
            missing_files.append(filename)

    print(f"Loaded {len(samples)} valid images with labels")
    if missing_files:
        print(f"Warning: {len(missing_files)} image files not found")

    return samples


def calculate_image_statistics(data_loader):
    """Calculate mean and standard deviation per channel for a dataset."""
    print("Calculating image statistics...")

    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    total_samples = 0

    for images, _ in data_loader:
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]

        # Reshape to (batch_size, -1, 3) for channel-wise stats
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(batch_size, -1, 3)

        batch_mean = batch_images.mean(axis=1)
        batch_std = batch_images.std(axis=1)

        total_mean += batch_mean.sum(axis=0)
        total_std += batch_std.sum(axis=0)
        total_samples += batch_size

    overall_mean = (total_mean / total_samples).tolist()
    overall_std = (total_std / total_samples).tolist()

    print(f"Image statistics: Mean={overall_mean}, Std={overall_std}")
    return {
        "mean": overall_mean,
        "std": overall_std,
        "total_samples": total_samples,
    }


class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.copy()

        if self.transform:
            image = self.transform(image)

        return image, label


def split_dataset(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Stratified split so both classes keep the same balance in every set."""
    print(f"\nSplitting dataset into train ({train_ratio}), val ({val_ratio}), test ({test_ratio})...")

    set_seed(random_seed)

    class_0 = [(path, label) for path, label in samples if label == 0]
    class_1 = [(path, label) for path, label in samples if label == 1]
    print(f"Class distribution before split: Class 0: {len(class_0)}, Class 1: {len(class_1)}")

    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        return train_data, val_data, test_data

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train_samples = train_0 + train_1
    val_samples = val_0 + val_1
    test_samples = test_0 + test_1

    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)
    np.random.shuffle(test_samples)

    print("Split complete:")
    print(f"  Training set: {len(train_samples)} images")
    print(f"  Validation set: {len(val_samples)} images")
    print(f"  Test set: {len(test_samples)} images")

    train_labels = [label for _, label in train_samples]
    val_labels = [label for _, label in val_samples]
    test_labels = [label for _, label in test_samples]
    print(f"  Train - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"  Val   - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
    print(f"  Test  - Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}")

    return train_samples, val_samples, test_samples


def create_dataloaders(train_samples, val_samples, test_samples, train_transforms, val_transforms, batch_size, num_workers=0):
    print(f"\nCreating DataLoaders with batch_size={batch_size}...")

    train_paths, train_labels = zip(*train_samples) if train_samples else ([], [])
    val_paths, val_labels = zip(*val_samples) if val_samples else ([], [])
    test_paths, test_labels = zip(*test_samples) if test_samples else ([], [])

    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)

    # num_workers must stay 0 to match the notebook: the augmentations draw from the
    # global torch generator, and dropout draws from it too. With worker processes the
    # augmentations would use per-worker generators, so the trained model would change.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
