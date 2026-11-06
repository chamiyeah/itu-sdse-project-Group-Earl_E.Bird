import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from botocore import UNSIGNED

from vial_classifier.data import acquire
from vial_classifier.data.acquire import (
    create_s3_client,
    download_all_images,
    download_files_from_metadata,
    download_one_file,
)

ENDPOINT = "https://example.invalid"
BUCKET = "test-bucket"


class FakeS3Client:
    """Replaces the boto3 client: records the calls and writes a small file, no network."""

    def __init__(self, failing_keys=()):
        self.calls = []
        self.failing_keys = failing_keys

    def download_file(self, bucket, key, filename):
        self.calls.append((bucket, key, filename))
        if key in self.failing_keys:
            raise RuntimeError("boom")
        Path(filename).write_bytes(b"downloaded")


def write_metadata(path, filenames):
    lines = ["filename,class"] + [f"{name},0" for name in filenames]
    path.write_text("\n".join(lines) + "\n")


def use_fake_client(monkeypatch, fake_client):
    """Make the module use fake_client and return the list of endpoints it was asked for."""
    requested_endpoints = []

    def fake_create_s3_client(endpoint_url):
        requested_endpoints.append(endpoint_url)
        return fake_client

    monkeypatch.setattr(acquire, "create_s3_client", fake_create_s3_client)
    return requested_endpoints


def test_requests_right_keys_and_saves_in_right_place(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "raw"
    write_metadata(metadata_path, ["a.jpg", "b.jpg"])
    fake_client = FakeS3Client()
    requested_endpoints = use_fake_client(monkeypatch, fake_client)

    download_files_from_metadata(metadata_path, output_dir, ENDPOINT, BUCKET, "images/raw/")

    assert requested_endpoints == [ENDPOINT]
    assert sorted(fake_client.calls) == [
        (BUCKET, "images/raw/a.jpg", str(output_dir / "a.jpg")),
        (BUCKET, "images/raw/b.jpg", str(output_dir / "b.jpg")),
    ]
    assert (output_dir / "a.jpg").read_bytes() == b"downloaded"
    assert (output_dir / "b.jpg").read_bytes() == b"downloaded"


def test_prefix_default_is_empty(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, ["a.jpg"])
    fake_client = FakeS3Client()
    use_fake_client(monkeypatch, fake_client)

    download_files_from_metadata(metadata_path, tmp_path / "raw", ENDPOINT, BUCKET)

    assert fake_client.calls[0][1] == "a.jpg"


def test_existing_file_is_skipped_and_not_overwritten(tmp_path, monkeypatch, capsys):
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    (output_dir / "a.jpg").write_bytes(b"already here")
    write_metadata(metadata_path, ["a.jpg", "b.jpg"])
    fake_client = FakeS3Client()
    use_fake_client(monkeypatch, fake_client)

    download_files_from_metadata(metadata_path, output_dir, ENDPOINT, BUCKET, "images/raw/")

    assert [call[1] for call in fake_client.calls] == ["images/raw/b.jpg"]
    assert (output_dir / "a.jpg").read_bytes() == b"already here"
    assert "Downloaded 1 images, skipped 1 existing, failed 0" in capsys.readouterr().out


def test_download_error_is_counted_and_others_continue(tmp_path, monkeypatch, capsys):
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "raw"
    write_metadata(metadata_path, ["a.jpg", "b.jpg", "c.jpg"])
    fake_client = FakeS3Client(failing_keys=["images/raw/b.jpg"])
    use_fake_client(monkeypatch, fake_client)

    download_files_from_metadata(metadata_path, output_dir, ENDPOINT, BUCKET, "images/raw/")

    output = capsys.readouterr().out
    assert "Failed to download images/raw/b.jpg: boom" in output
    assert "Downloaded 2 images, skipped 0 existing, failed 1" in output
    assert (output_dir / "a.jpg").exists()
    assert not (output_dir / "b.jpg").exists()
    assert (output_dir / "c.jpg").exists()


def test_missing_metadata_raises_clear_error_before_creating_client(tmp_path, monkeypatch):
    requested_endpoints = use_fake_client(monkeypatch, FakeS3Client())

    with pytest.raises(FileNotFoundError, match="run `dvc pull` first"):
        download_files_from_metadata(tmp_path / "missing.csv", tmp_path / "raw", ENDPOINT, BUCKET)

    assert requested_endpoints == []


def test_uses_eight_threads(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path, ["a.jpg"])
    use_fake_client(monkeypatch, FakeS3Client())
    requested_workers = []

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers):
            requested_workers.append(max_workers)
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(acquire, "ThreadPoolExecutor", RecordingExecutor)

    download_files_from_metadata(metadata_path, tmp_path / "raw", ENDPOINT, BUCKET)

    assert requested_workers == [8]


def test_download_one_file_return_values(tmp_path):
    fake_client = FakeS3Client(failing_keys=["bad.jpg"])
    existing = tmp_path / "existing.jpg"
    existing.write_bytes(b"already here")

    assert download_one_file(fake_client, BUCKET, "good.jpg", tmp_path / "good.jpg") is True
    assert download_one_file(fake_client, BUCKET, "existing.jpg", existing) is False
    assert download_one_file(fake_client, BUCKET, "bad.jpg", tmp_path / "bad.jpg") is False


def test_download_all_images_fetches_images_and_test_set(tmp_path, monkeypatch):
    image_metadata = tmp_path / "images.csv"
    test_metadata = tmp_path / "test.csv"
    write_metadata(image_metadata, ["train.jpg"])
    write_metadata(test_metadata, ["test.jpg"])
    fake_client = FakeS3Client()
    requested_endpoints = use_fake_client(monkeypatch, fake_client)

    download_all_images(
        ENDPOINT,
        BUCKET,
        image_metadata,
        tmp_path / "raw",
        "images/raw/",
        test_metadata,
        tmp_path / "test",
        "test/",
    )

    assert requested_endpoints == [ENDPOINT, ENDPOINT]
    assert fake_client.calls == [
        (BUCKET, "images/raw/train.jpg", str(tmp_path / "raw" / "train.jpg")),
        (BUCKET, "test/test.jpg", str(tmp_path / "test" / "test.jpg")),
    ]


def test_create_s3_client_is_unsigned_and_uses_no_network(monkeypatch):
    connections = []

    def record_connection(self, address):
        connections.append(address)
        raise OSError("network is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", record_connection)

    client = create_s3_client(ENDPOINT)

    assert client.meta.config.signature_version is UNSIGNED
    assert client.meta.endpoint_url == ENDPOINT
    assert connections == []
