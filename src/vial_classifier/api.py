"""Minimal model-serving API for the smoke pipeline."""

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from torchvision import transforms

from vial_classifier.artifacts import load_model_artifact
from vial_classifier.config import get_settings

app = FastAPI(title="Vial Classifier")
settings = get_settings()


def _predict(payload: bytes) -> dict[str, object]:
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc
    model, metadata = load_model_artifact(Path(settings.artifact_dir))
    transform = transforms.Compose(
        [
            transforms.Resize(tuple(metadata["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    with torch.no_grad():
        probability = torch.sigmoid(model(transform(image).unsqueeze(0))).item()
    threshold = float(metadata["threshold"])
    label = int(probability >= threshold)
    return {
        "prediction": label,
        "class_name": metadata["class_names"][label],
        "confidence": probability if label else 1 - probability,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: Request) -> dict[str, object]:
    return _predict(await request.body())


@app.get("/drift")
def drift() -> dict[str, object]:
    return {"overall_drift_detected": False, "reason": "No inference log configured"}
