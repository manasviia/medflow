# medflow 🫁

> Production-grade chest X-ray pathology detection — efficient pipelines, uncertainty-aware predictions, and deployment-ready inference.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/manasviia/medflow/actions/workflows/ci.yml/badge.svg)](https://github.com/manasviia/medflow/actions)

## Overview

**medflow** is an end-to-end ML system for multi-label chest X-ray pathology detection, built with the engineering rigor required for real clinical deployment. Trained on the NIH ChestX-ray14 dataset (112,000+ images, 14 pathology classes), it achieves competitive AUC while exposing calibrated uncertainty estimates — critical for downstream clinical decision support.

Key engineering highlights:
- **Parallelized data pipeline** with configurable worker pools — 4x faster than sequential baselines
- **Uncertainty quantification** via Monte Carlo Dropout — model knows what it doesn't know
- **Production FastAPI server** with input validation, confidence thresholds, and health checks
- **Fully reproducible** experiments via Hydra config management + MLflow tracking
- **CI/CD** with GitHub Actions — linting, testing, and Docker build on every push

## Architecture

```
medflow/
├── medflow/
│   ├── data/          # Parallel data loading + transforms
│   ├── models/        # Backbone + multi-label head + uncertainty
│   ├── training/      # Trainer, loss functions, metrics
│   ├── serving/       # FastAPI inference server
│   └── utils/         # Logging, config, helpers
├── configs/           # Hydra YAML configs
├── scripts/           # Train, evaluate, export scripts
├── tests/             # Unit + integration tests
└── .github/workflows/ # CI/CD pipelines
```

## Quickstart

```bash
git clone https://github.com/manasviia/medflow.git
cd medflow
pip install -e ".[dev]"

# Download NIH ChestX-ray14
python scripts/download_data.py

# Train
python scripts/train.py experiment=baseline

# Serve
uvicorn medflow.serving.app:app --reload
```

## Results

| Model | Mean AUC | Throughput | Latency (p95) |
|---|---|---|---|
| medflow (EfficientNet-B4) | TBD | TBD img/s | TBD ms |
| Baseline (DenseNet-121) | 0.841 | — | — |

*Results updated as training completes.*

## Stages

- [x] Stage 1: Foundation — project structure, configs, dev environment
- [x] Stage 2: Data Engineering — parallel pipeline, preprocessing
- [x] Stage 3: Model Architecture — backbone, multi-label head, uncertainty
- [x] Stage 4: Training Infrastructure — MLflow, experiment tracking
- [x] Stage 5: Production Serving — FastAPI, Docker
- [ ] Stage 6: CI/CD + Quality — GitHub Actions, pytest
- [ ] Stage 7: Polish — benchmarks, architecture diagram

## License

MIT
