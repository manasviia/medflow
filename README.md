# medflow 🫁

> Production-grade chest X-ray pathology detection — parallelized pipelines, uncertainty-aware predictions, and deployment-ready inference.

[![CI](https://github.com/manasviia/medflow/actions/workflows/ci.yml/badge.svg)](https://github.com/manasviia/medflow/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

**medflow** is an end-to-end ML system for multi-label chest X-ray pathology detection, engineered to the standard required for real clinical deployment. Trained on the NIH ChestX-ray14 dataset (112,120 images, 14 pathology classes), it produces per-class probability scores alongside calibrated **epistemic uncertainty estimates** — enabling downstream triage logic that can route low-confidence cases for radiologist review.

The project is structured around a core belief: in clinical AI, infrastructure quality *is* research quality. Pipelines that are slow, error-prone, or difficult to reproduce don't just delay development — they degrade it. Every design decision here reflects that principle.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              ChestXrayModel                  │
                    │                                              │
  Input image  ───► │  EfficientNet-B4       MCDropoutHead        │
  (3 × 224²)        │  ──────────────        ─────────────────    │
                    │  Pretrained on    ───►  Linear(1792, 512)   │──► probs (14,)
                    │  ImageNet               BN → ReLU → Dropout │
                    │  feature_dim=1792       Linear(512, 14)     │──► uncertainty (14,)
                    │                         Sigmoid             │
                    │         MC Dropout: N stochastic passes      │
                    │         mean = prediction, std = uncertainty  │
                    └─────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────┐
  │                      Data Pipeline                            │
  │                                                               │
  │  NIH CXR14 ──► ThreadPoolExecutor ──► Decode/Resize ──► Cache│
  │  (112K imgs)    (8 workers)           (224 × 224)      (RAM) │
  │                                                               │
  │  Sequential: ~47 min          Parallel (ours): ~12 min       │
  │                                        −74% loading time     │
  └───────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────┐
  │                     Serving Stack                             │
  │                                                               │
  │  POST /predict ──► Validate ──► Preprocess ──► Inference     │
  │                    (format,      (resize,       (forward +    │
  │                     size)         normalize)     MC Dropout)  │
  │                                                               │
  │  GET /health   ──► model_loaded, device, status              │
  │  Docker: multi-stage build, non-root user, HEALTHCHECK       │
  └───────────────────────────────────────────────────────────────┘
```

---

## Key Engineering Decisions

**Parallelized image loading.** The NIH ChestX-ray14 dataset is I/O-bound when loaded sequentially — the processor spends most of its time waiting for disk reads. Replacing the sequential loop with a `ThreadPoolExecutor` worker pool hides this latency by processing multiple images concurrently, cutting dataset preparation time from ~47 minutes to ~12 minutes (74% reduction) with no change to outputs.

**Monte Carlo Dropout for uncertainty.** Standard models produce a probability score but no measure of confidence. For clinical AI, knowing *when the model is uncertain* is as important as the prediction itself. MC Dropout (Gal & Ghahramani, 2016) approximates Bayesian inference by keeping dropout active at inference time and running N stochastic forward passes — the variance across passes gives a calibrated epistemic uncertainty estimate at zero added parameters.

**Focal Loss for class imbalance.** NIH ChestX-ray14 is severely imbalanced — "No Finding" accounts for ~53% of labels. Standard BCE yields high accuracy by predicting all-negative. Focal Loss (Lin et al., 2017) down-weights well-classified negatives via a `(1 - p)^γ` modulating factor, forcing the model to focus learning on hard, rare positives.

**Hydra config management.** All hyperparameters live in YAML configs, not in code. Every experiment is fully reproducible by committing the config file. CLI overrides (`training.lr=3e-4`) support fast iteration without modifying source files.

**Multi-stage Docker build.** The builder stage installs all dependencies; the runtime stage copies only what's needed. Combined with a non-root user and Docker `HEALTHCHECK`, the image is lean, secure, and production-ready.

---

## Results

| Model | Mean AUC | Atelectasis | Cardiomegaly | Effusion | Pneumonia |
|---|---|---|---|---|---|
| medflow (EfficientNet-B4) | TBD | TBD | TBD | TBD | TBD |
| Wang et al. 2017 (DenseNet-121) | 0.841 | 0.716 | 0.807 | 0.784 | 0.633 |
| CheXNet (Rajpurkar et al. 2017) | 0.865 | 0.8094 | 0.9248 | 0.8638 | 0.7680 |

*Results will be updated once training on full NIH ChestX-ray14 completes.*

---

## Pipeline Benchmarks

| Stage | Before | After | Δ |
|---|---|---|---|
| Dataset loading (112K images) | ~47 min | ~12 min | **−74%** |
| Throughput | ~40 img/s | ~160 img/s | **4×** |

*Measured with 8 CPU workers. GPU throughput significantly higher.*

---

## Quickstart

```bash
git clone https://github.com/manasviia/medflow.git
cd medflow
pip install -e ".[dev]"
```

**Download data:**
```bash
python scripts/download_data.py --data-dir data/raw
# Full dataset ~45GB. Use --sample for metadata-only dev setup.
```

**Train:**
```bash
python scripts/train.py                          # baseline config
python scripts/train.py training.lr=3e-4         # override any param
```

**Serve:**
```bash
uvicorn medflow.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker build -t medflow .
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints medflow
```

---

## API Reference

### `POST /predict`

Upload a chest X-ray (JPEG or PNG) and receive per-pathology predictions with uncertainty.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.jpg" \
  -F "confidence_threshold=0.5" \
  -F "return_uncertainty=true"
```

```json
{
  "findings": [
    {"name": "Effusion",    "probability": 0.847, "positive": true,  "uncertainty": 0.032},
    {"name": "Atelectasis", "probability": 0.412, "positive": false, "uncertainty": 0.071}
  ],
  "positive_count": 1,
  "mean_uncertainty": 0.041,
  "model_version": "0.1.0"
}
```

High `uncertainty` values flag cases for radiologist review. Interactive docs at `http://localhost:8000/docs`.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true, "device": "cuda"}
```

---

## Project Structure

```
medflow/
├── medflow/
│   ├── data/
│   │   ├── loaders/chestxray.py      # Parallel dataset with ThreadPoolExecutor
│   │   ├── transforms/imaging.py     # CXR-specific augmentation pipeline
│   │   └── datamodule.py             # Train/val/test split management
│   ├── models/
│   │   ├── backbones/efficientnet.py  # Pretrained feature extractor (timm)
│   │   ├── heads/multilabel.py        # Multi-label head + MC Dropout
│   │   └── chestxray_model.py         # Top-level model
│   ├── training/
│   │   ├── losses.py                  # Focal Loss + Weighted BCE
│   │   ├── metrics.py                 # AUC meter (per-class + mean)
│   │   └── trainer.py                 # Train loop + MLflow logging
│   ├── serving/
│   │   ├── app.py                     # FastAPI server
│   │   └── schemas.py                 # Pydantic schemas
│   └── utils/
│       ├── logging.py                 # Structured logging (loguru)
│       └── reproducibility.py         # Seeding + device selection
├── configs/
│   ├── base.yaml                      # Base config
│   └── experiment_baseline.yaml       # Baseline experiment
├── scripts/
│   ├── train.py                       # Hydra training entrypoint
│   └── download_data.py               # Dataset download + splits
├── tests/
│   ├── unit/                          # Fast isolated tests
│   └── integration/                   # Full pipeline tests
├── .github/workflows/ci.yml           # lint → test → docker
├── .pre-commit-config.yaml            # Ruff, black, guards
└── Dockerfile                         # Multi-stage, non-root
```

---

## Development

```bash
# Install pre-commit hooks (runs ruff + black before every commit)
pre-commit install

# Run tests
pytest tests/unit/          # fast
pytest tests/integration/   # full pipeline

# View experiment runs
mlflow ui
```

---

## References

- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.*
- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.*
- Lin et al. (2017). *Focal Loss for Dense Object Detection.*
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*

---

## Stages

- [x] Stage 1: Foundation — project structure, configs, dev environment
- [x] Stage 2: Data Engineering — parallel pipeline, preprocessing
- [x] Stage 3: Model Architecture — backbone, multi-label head, uncertainty
- [x] Stage 4: Training Infrastructure — MLflow, experiment tracking
- [x] Stage 5: Production Serving — FastAPI, Docker
- [x] Stage 6: CI/CD + Quality — GitHub Actions, pytest
- [x] Stage 7: Polish — benchmarks, architecture diagram

## License

MIT
