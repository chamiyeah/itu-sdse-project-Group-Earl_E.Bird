import torch

from vial_classifier.artifacts import save_model_artifact, validate_model_artifact
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
