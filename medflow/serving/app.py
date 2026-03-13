"""FastAPI inference server for medflow.

Exposes two endpoints:
  POST /predict  — upload a chest X-ray image, get pathology predictions
  GET  /health   — liveness check (used by Docker/k8s health probes)

Design decisions:
- Model is loaded once at startup via lifespan context (not per-request)
- Input images are validated for format and size before reaching the model
- MC Dropout uncertainty is optional (adds ~20x inference time, useful for
  flagging low-confidence cases for radiologist review)
- All errors return structured JSON with a clear message field

Usage:
    uvicorn medflow.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from PIL import Image, UnidentifiedImageError

from medflow.data.transforms.imaging import get_val_transforms
from medflow.models.chestxray_model import ChestXrayModel
from medflow.serving.schemas import (
    HealthResponse,
    PathologyResult,
    PredictionResponse,
)
from medflow.utils.reproducibility import get_device

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
IMAGE_SIZE = 224
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# ── App state ─────────────────────────────────────────────────────────────────

_model: ChestXrayModel | None = None
_device: torch.device | None = None
_transform = get_val_transforms(IMAGE_SIZE)


# ── Lifespan: load model once at startup ──────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _model, _device

    _device = get_device()
    _model = ChestXrayModel(pretrained=False)  # weights loaded from checkpoint

    checkpoint_path = CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(
            f"No checkpoint found at {checkpoint_path}. "
            "Running with random weights (for dev/testing only)."
        )

    _model = _model.to(_device)
    _model.eval()
    logger.info(f"Model ready on {_device}")

    yield  # Server is running

    logger.info("Shutting down — releasing model")
    _model = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="medflow",
    description="Production chest X-ray pathology detection with uncertainty quantification.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health() -> HealthResponse:
    """Liveness check. Returns 200 if the server is up and model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=str(_device) if _device else "unknown",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG or PNG)"),
    confidence_threshold: float = 0.5,
    return_uncertainty: bool = True,
) -> PredictionResponse:
    """Run pathology detection on a chest X-ray image.

    Returns per-class probabilities, positive/negative flags, and optional
    uncertainty estimates from Monte Carlo Dropout.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Validate file ──────────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(raw_bytes) / 1e6:.1f} MB). Max 10 MB.",
        )

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    tensor = _transform(image).unsqueeze(0).to(_device)  # (1, 3, H, W)

    # ── Inference ─────────────────────────────────────────────────────────────
    if return_uncertainty:
        mean_probs, uncertainty = _model.predict(tensor)
        probs = mean_probs.squeeze(0).tolist()
        unc = uncertainty.squeeze(0).tolist()
        mean_unc = float(sum(unc) / len(unc))
    else:
        with torch.no_grad():
            probs = _model(tensor).squeeze(0).tolist()
        unc = [None] * len(PATHOLOGY_CLASSES)
        mean_unc = None

    # ── Build response ────────────────────────────────────────────────────────
    findings = [
        PathologyResult(
            name=name,
            probability=round(float(prob), 4),
            positive=float(prob) >= confidence_threshold,
            uncertainty=round(float(u), 4) if u is not None else None,
        )
        for name, prob, u in zip(PATHOLOGY_CLASSES, probs, unc)
    ]

    return PredictionResponse(
        findings=findings,
        positive_count=sum(f.positive for f in findings),
        mean_uncertainty=round(mean_unc, 4) if mean_unc is not None else None,
        model_version=MODEL_VERSION,
    )
