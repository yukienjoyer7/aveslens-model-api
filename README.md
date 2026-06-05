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

Bird species classifier — 525 classes, Swin Tiny + ONNX, served as a FastAPI inference API.

## Endpoint

```
POST /predict
Content-Type: multipart/form-data

file: <image>
```

```json
{
  "predictions": [
    { "label": "AFRICAN CROWNED CRANE", "confidence": 0.94 },
    { "label": "GREY CROWNED CRANE", "confidence": 0.04 }
  ]
}
```

## Local dev

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
