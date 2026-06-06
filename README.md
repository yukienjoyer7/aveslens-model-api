---
title: AvesLens Model API
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# AvesLens Model API

![CI](https://github.com/crtal7/aveslens-model-api/actions/workflows/ci.yml/badge.svg)

Bird species classifier — 525 classes, Swin Tiny + ONNX, served as a FastAPI inference API.

**Base URL:** `https://crtal-aveslens-model-api.hf.space`

---

## Endpoints

### `GET /health`

Check if the API is running.

```bash
curl https://crtal-aveslens-model-api.hf.space/health
```

```json
{ "status": "ok" }
```

---

### `POST /predict`

Classify a bird image. Returns the top 5 predicted species with confidence scores.

**Request**

| Field | Type | Description |
|---|---|---|
| `file` | `multipart/form-data` | Image file (JPEG, PNG, WEBP) |

**Response**

```json
{
  "predictions": [
    { "label": "AFRICAN CROWNED CRANE", "confidence": 0.94 },
    { "label": "GREY CROWNED CRANE", "confidence": 0.04 },
    { "label": "DEMOISELLE CRANE", "confidence": 0.01 }
  ]
}
```

**Examples**

cURL:
```bash
curl -X POST https://crtal-aveslens-model-api.hf.space/predict \
  -F "file=@bird.jpg"
```

Python:
```python
import requests

with open("bird.jpg", "rb") as f:
    response = requests.post(
        "https://crtal-aveslens-model-api.hf.space/predict",
        files={"file": ("bird.jpg", f, "image/jpeg")},
    )

print(response.json())
```

Android (OkHttp):
```kotlin
val file = File("bird.jpg")
val requestBody = MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", file.name, file.asRequestBody("image/jpeg".toMediaType()))
    .build()

val request = Request.Builder()
    .url("https://crtal-aveslens-model-api.hf.space/predict")
    .post(requestBody)
    .build()

val response = OkHttpClient().newCall(request).execute()
println(response.body?.string())
```

---

## Model

| Property | Value |
|---|---|
| Architecture | Swin Tiny (`swin_tiny_patch4_window7_224`) |
| Classes | 525 bird species |
| Input | 224 × 224 RGB image |
| Format | ONNX (CPU inference) |
| Top-1 accuracy | ~97% on validation set |
| Weights | [`crtal/swin-tiny-bird525-onnx`](https://huggingface.co/crtal/swin-tiny-bird525-onnx) |

---

## Local dev

```bash
pip install -r requirements.txt
HF_MODEL_REPO=crtal/swin-tiny-bird525 uvicorn app.main:app --reload
```
