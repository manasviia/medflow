"""Pydantic request/response schemas for the inference API.

Explicit schemas serve two purposes:
1. Input validation — FastAPI rejects malformed requests before they hit the model
2. Self-documenting API — schemas are auto-rendered in /docs (Swagger UI)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for single-image inference."""

    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum probability to flag a pathology as positive.",
    )
    return_uncertainty: bool = Field(
        default=True,
        description="If true, run MC Dropout and return uncertainty estimates.",
    )


class PathologyResult(BaseModel):
    """Prediction result for a single pathology class."""

    name: str
    probability: float = Field(ge=0.0, le=1.0)
    positive: bool
    uncertainty: float | None = Field(
        default=None,
        description="Epistemic uncertainty (std across MC Dropout samples). "
        "Higher = model is less confident.",
    )


class PredictionResponse(BaseModel):
    """Full inference response."""

    findings: list[PathologyResult]
    positive_count: int = Field(description="Number of flagged pathologies.")
    mean_uncertainty: float | None = Field(
        default=None,
        description="Mean uncertainty across all classes (if MC Dropout enabled).",
    )
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
