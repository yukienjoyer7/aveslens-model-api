import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = os.environ.get("HF_MODEL_REPO", "crtal/swin-tiny-bird525-onnx")
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


class BirdClassifier:
    def __init__(self) -> None:
        onnx_path = hf_hub_download(repo_id=REPO_ID, filename="model.onnx")
        labels_path = hf_hub_download(repo_id=REPO_ID, filename="labels.json")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.labels: dict[str, str] = json.loads(Path(labels_path).read_text())
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        logits = self.session.run(None, {self.input_name: _preprocess(image)})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top_indices = probs.argsort()[::-1][:top_k]
        return [
            {"label": self.labels[str(i)], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]


classifier: BirdClassifier | None = None


def get_classifier() -> BirdClassifier:
    global classifier
    if classifier is None:
        classifier = BirdClassifier()
    return classifier
