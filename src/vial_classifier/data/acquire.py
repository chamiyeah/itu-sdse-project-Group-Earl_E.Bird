import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig


def create_s3_client(endpoint_url):
    """Create an anonymous boto3 S3 client for a public bucket."""
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        config=BotoConfig(signature_version=UNSIGNED)
    )


def download_one_file(s3_client, bucket, key, output_path):
    """Download a single file from S3 using boto3."""
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(bucket, key, str(output_path))
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False


def download_files_from_metadata(metadata_path, output_dir, endpoint_url, bucket, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    if not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            "It is not stored in Git: run `dvc pull` first."
        )

    s3_client = create_s3_client(endpoint_url)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(download_one_file, s3_client, bucket, key, output_path)
            futures[future] = (key, output_path)

        for future in as_completed(futures):
            key, output_path = futures[future]
            if future.result():
                downloaded_count += 1
            else:
                if output_path.exists():
                    skipped_count += 1
                else:
                    failed_count += 1

    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")


def download_all_images(
    endpoint_url,
    bucket,
    image_metadata_path,
    image_data_dir,
    images_prefix,
    test_metadata_path,
    test_image_dir,
    test_prefix,
):
    """Pull the training and test images listed in the metadata from the object store."""
    download_files_from_metadata(image_metadata_path, image_data_dir, endpoint_url, bucket, images_prefix)
    download_files_from_metadata(test_metadata_path, test_image_dir, endpoint_url, bucket, test_prefix)
