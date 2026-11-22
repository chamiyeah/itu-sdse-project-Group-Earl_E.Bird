import json

import torch

from vial_classifier.artifacts import (
    ARTIFACT_METADATA_FILE,
    save_model_artifact,
    validate_model_artifact,
)
from vial_classifier.models.cnn import SimpleCNN


def test_model_artifact_can_be_saved_and_validated(tmp_path):
    artifact_dir = save_model_artifact(
        SimpleCNN(),
        tmp_path,
        threshold=0.5,
        image_size=(64, 64),
        metrics={"test_f1": 0.5},
    )

    result = validate_model_artifact(artifact_dir)

    assert result["valid"] is True
    assert result["output_shape"] == [1, 1]
    assert (artifact_dir / "vial_cnn_model.pth").exists()


def test_training_checkpoint_dictionary_can_be_validated(tmp_path):
    model = SimpleCNN()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "SimpleCNN",
            "input_shape": (3, 64, 64),
            "output_shape": (1,),
            "threshold": 0.3,
            "num_classes": 2,
        },
        tmp_path / "vial_cnn_model.pth",
    )
    (tmp_path / ARTIFACT_METADATA_FILE).write_text(json.dumps({
        "model_type": "SimpleCNN",
        "checkpoint": "vial_cnn_model.pth",
        "class_names": ["clean", "defective"],
        "threshold": 0.3,
        "image_size": [64, 64],
        "metrics": {},
    }))

    result = validate_model_artifact(tmp_path)

    assert result["valid"] is True
    assert result["output_shape"] == [1, 1]
