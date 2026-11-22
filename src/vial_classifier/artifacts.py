"""Model artifact persistence and validation."""

import json
from pathlib import Path
from typing import Any

import torch

from vial_classifier.models.cnn import SimpleCNN

ARTIFACT_METADATA_FILE = "model_metadata.json"


def save_model_artifact(
    model: SimpleCNN,
    output_dir: Path,
    *,
    threshold: float,
    image_size: tuple[int, int],
    metrics: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "vial_cnn_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    metadata = {
        "model_type": "SimpleCNN",
        "checkpoint": checkpoint_path.name,
        "class_names": ["clean", "defective"],
        "threshold": threshold,
        "image_size": list(image_size),
        "metrics": metrics,
    }
    (output_dir / ARTIFACT_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(metadata["class_names"], indent=2) + "\n"
    )
    return output_dir


def load_model_artifact(
    artifact_dir: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SimpleCNN, dict[str, Any]]:
    metadata_path = artifact_dir / ARTIFACT_METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    checkpoint_path = artifact_dir / metadata["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = SimpleCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def validate_model_artifact(artifact_dir: Path) -> dict[str, Any]:
    model, metadata = load_model_artifact(artifact_dir)
    image_size = tuple(metadata["image_size"])
    if image_size != (64, 64):
        raise ValueError(f"Unsupported image size in artifact: {image_size}")
    with torch.no_grad():
        output = model(torch.zeros((1, 3, *image_size)))
    if tuple(output.shape) != (1, 1):
        raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")
    return {
        "valid": True,
        "checkpoint": metadata["checkpoint"],
        "output_shape": list(output.shape),
    }
