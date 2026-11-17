"""Small fake dataset shared by the pipeline tests. No real data, no network."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# 20 clean + 40 defective. The split takes int(70%) / int(15%) / the rest of every
# class, so clean gives 14/3/3 and defective 28/6/6: train, val and test all contain
# both classes (ROC-AUC and the loss weight need that), and the 42 training images
# make two batches of 32, so the shuffle of the training loader really matters.
NUM_CLEAN = 20
NUM_DEFECTIVE = 40


def make_synthetic_dataset(root, seed=0):
    """Write data/images/metadata.csv and data/images/raw/*.jpg under root.

    The csv has only the two columns that load_image_dataset reads: filename, class.
    Images come from a local generator (never the global NumPy one), so creating
    the dataset does not change the random state of the code under test.
    """
    rng = np.random.default_rng(seed)
    raw_dir = Path(root) / "data" / "images" / "raw"
    raw_dir.mkdir(parents=True)

    rows = []
    for label, count in [(0, NUM_CLEAN), (1, NUM_DEFECTIVE)]:
        for index in range(count):
            filename = f"vial_{label}_{index:03d}.jpg"
            # Defective images are brighter on average, so the model has something
            # to learn. Every image also gets its own random brightness shift, so
            # the two classes overlap and the metrics do not all come out as 1.0.
            brightness = (110 if label == 0 else 135) + rng.normal(0, 25)
            pixels = rng.normal(brightness, 40, size=(24, 32, 3))
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(raw_dir / filename, format="JPEG")
            rows.append({"filename": filename, "class": label})

    with open(Path(root) / "data" / "images" / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class"])
        writer.writeheader()
        writer.writerows(rows)
