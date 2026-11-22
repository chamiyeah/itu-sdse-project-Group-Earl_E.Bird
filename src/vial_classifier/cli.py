import argparse
import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from vial_classifier.artifacts import validate_model_artifact
from vial_classifier.config import get_settings
from vial_classifier.data.acquire import download_all_images
from vial_classifier.pipeline import run_training


def run_acquire():
    """Download the training and test images, using the values in Config."""
    config = get_settings()
    download_all_images(
        endpoint_url=config.s3_endpoint,
        bucket=config.s3_bucket,
        image_metadata_path=config.metadata_path,
        image_data_dir=config.image_data_dir,
        images_prefix=config.s3_images_prefix,
        test_metadata_path=config.test_metadata_path,
        test_image_dir=config.test_image_dir,
        test_prefix=config.s3_test_prefix,
    )


def create_smoke_dataset(root):
    """Create a small local dataset so Dagger smoke training needs no network."""
    rng = np.random.default_rng(0)
    raw_dir = Path(root) / "data" / "images" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, count in [(0, 20), (1, 40)]:
        for index in range(count):
            filename = f"vial_{label}_{index:03d}.jpg"
            brightness = (110 if label == 0 else 135) + rng.normal(0, 25)
            pixels = rng.normal(brightness, 40, size=(24, 32, 3))
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(raw_dir / filename, format="JPEG")
            rows.append({"filename": filename, "class": label})

    metadata_path = Path(root) / "data" / "images" / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=["filename", "class"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    """Run one sub-command and return the exit code (0 = ok)."""
    parser = argparse.ArgumentParser(prog="vial_classifier", description="Vial classifier training job")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("acquire", help="download the images listed in the metadata files")
    subparsers.add_parser("test", help="run the Python test suite")
    subparsers.add_parser("validate", help="validate the model artifact")

    train_parser = subparsers.add_parser("train", help="train the model and write the artifacts")
    train_parser.add_argument(
        "--mode",
        choices=["full", "smoke"],
        default="full",
        help="use a small local dataset for smoke training",
    )

    # argparse prints the error and raises SystemExit(2) when the command is
    # missing or unknown. We turn that into a return value, so main() always
    # returns an exit code instead of exiting by itself.
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return error.code

    if args.command == "acquire":
        run_acquire()
    elif args.command == "test":
        return subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False).returncode
    elif args.command == "train":
        config = get_settings()
        if args.mode == "smoke":
            create_smoke_dataset(Path.cwd())
            config = replace(
                config,
                num_epochs=1,
                batch_size=32,
                image_data_dir="./data/images/raw",
                metadata_path="./data/images/metadata.csv",
            )
        run_training(config)
    elif args.command == "validate":
        config = get_settings()
        print(validate_model_artifact(Path(config.artifact_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
