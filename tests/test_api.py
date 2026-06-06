import io
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg_bytes() -> bytes:
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MOCK_PREDICTIONS = [
    {"label": "AMERICAN ROBIN", "confidence": 0.95},
    {"label": "HOUSE SPARROW", "confidence": 0.03},
]


@patch("app.model.get_classifier")
def test_predict_returns_predictions(mock_get_classifier):
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = MOCK_PREDICTIONS
    mock_get_classifier.return_value = mock_classifier

    from app.main import app
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("bird.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert body["predictions"][0]["label"] == "AMERICAN ROBIN"
    assert body["predictions"][0]["confidence"] == 0.95


@patch("app.model.get_classifier")
def test_health(_):
    from app.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.model.get_classifier")
def test_predict_rejects_non_image(_):
    from app.main import app
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("data.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
