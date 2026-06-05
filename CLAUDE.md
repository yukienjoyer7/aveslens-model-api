# AvesLens Model API

Bird species classifier (525 classes) built with Swin Tiny, served as a FastAPI inference API, deployed to HF Spaces via GitHub Actions CI/CD.

## Stack

- **Model**: `swin_tiny_patch4_window7_224` via `timm`, exported to ONNX for CPU inference
- **Dataset**: [`yashikota/birds-525-species-image-classification`](https://huggingface.co/datasets/yashikota/birds-525-species-image-classification) — 89,885 images, 525 classes, splits: `train` (84,600) / `validation` (2,630) / `test` (2,630), columns: `image` (PIL), `label` (int 0–524)
- **API**: FastAPI + ONNX Runtime, port 7860
- **Deployment**: HF Spaces (Docker runtime), weights hosted separately on HF Hub
- **CI/CD**: GitHub Actions — lint/test on PR, sync to HF Space on merge to main

## Project Structure

```
aveslens-model-api/
├── app/
│   ├── main.py            ← FastAPI app, health check + /predict
│   ├── model.py           ← ONNX session, preprocessing, inference
│   └── schemas.py         ← Request/response Pydantic types
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── config.yaml        ← All hyperparameters live here
├── .github/
│   └── workflows/
│       ├── ci.yml         ← Lint + unit tests on PR/push
│       └── deploy.yml     ← Sync to HF Space on merge to main
├── Dockerfile
├── requirements.txt
└── README.md              ← Doubles as HF Space metadata (sdk: docker)
```

## API

```
POST /predict
  body: multipart image upload
  response: { "predictions": [{ "label": "...", "confidence": 0.94 }, ...] }
```

## Model weights

Stored on HF Hub (separate repo from the Space). Loaded at container startup via `huggingface_hub.hf_hub_download`. Update weights independently of API code.

## Secrets (GitHub)

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Push to HF Space + pull model weights from HF Hub |

---

## Build Phases

### Phase 1 — Repo Bootstrap
```
chore: initialize project structure with placeholder modules
chore: add .gitignore for python, model weights, and env files
docs: add README with project overview and setup instructions
chore: add requirements.txt with core dependencies (torch, timm, fastapi, onnxruntime)
```

### Phase 2 — Data Pipeline
```
feat(data): add bird525 dataset loader with huggingface datasets
feat(data): add train/val/test split handling and label map export
feat(data): add image preprocessing pipeline (resize, normalize imagenet stats)
feat(data): add training augmentations (flip, color jitter, random crop)
```

### Phase 3 — Model Definition
```
feat(model): add swin_tiny backbone loader via timm with pretrained weights
feat(model): replace classifier head for 525-class bird output
feat(model): add two-stage training strategy (frozen backbone → full finetune)
```

### Phase 4 — Training Loop
```
feat(training): add training loop with crossentropy loss and adamw optimizer
feat(training): add cosine annealing scheduler with warmup
feat(training): add top-1 and top-5 accuracy tracking per epoch
feat(training): add best checkpoint saving on val accuracy improvement
feat(training): add config.yaml for all hyperparameters
```

### Phase 5 — Evaluation + Export
```
feat(eval): add evaluation script with per-class accuracy report
feat(export): add onnx export script with dynamic input shape
feat(export): add labels.json export for inference class mapping
chore(export): add model upload script to huggingface hub
```

### Phase 6 — FastAPI App
```
feat(api): add fastapi app skeleton with health check endpoint
feat(api): add POST /predict endpoint accepting multipart image upload
feat(api): add onnx inference runner with imagenet preprocessing
feat(api): add response schema with top-k predictions and confidence scores
feat(api): load model weights from hf hub on container startup
test(api): add unit tests for predict endpoint with mock image input
```

### Phase 7 — Docker
```
chore(docker): add dockerfile for fastapi app on port 7860
chore(docker): add .dockerignore to exclude training artifacts and cache
fix(docker): pin python and onnxruntime versions for reproducible builds
```

### Phase 8 — CI/CD
```
ci: add ci workflow for linting and unit tests on pull requests
ci: add deploy workflow to sync repo to hf space on push to main
ci: add hf_token secret reference in deploy workflow
docs: add github actions badge to README
```

### Hotfix / maintenance patterns
```
fix(api): handle non-square image inputs without distortion
fix(model): correct label index off-by-one in prediction response
perf(api): reduce onnx session init time by caching at module level
chore(deps): bump onnxruntime to 1.x.x for arm64 compatibility
```
