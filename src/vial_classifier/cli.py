import argparse

from vial_classifier.config import Config
from vial_classifier.data.acquire import download_all_images
from vial_classifier.pipeline import run_training


def run_acquire():
    """Download the training and test images, using the values in Config."""
    config = Config()
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


def main(argv=None):
    """Run one sub-command and return the exit code (0 = ok)."""
    parser = argparse.ArgumentParser(prog="vial_classifier", description="Vial classifier training job")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("acquire", help="download the images listed in the metadata files")
    subparsers.add_parser("train", help="train the model and write the artifacts")

    # argparse prints the error and raises SystemExit(2) when the command is
    # missing or unknown. We turn that into a return value, so main() always
    # returns an exit code instead of exiting by itself.
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return error.code

    if args.command == "acquire":
        run_acquire()
    elif args.command == "train":
        run_training(Config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
