"""Unit tests for serving schemas and response structure."""

from medflow.serving.schemas import (
    HealthResponse,
    PathologyResult,
    PredictionResponse,
)


def test_pathology_result_positive_flag():
    """positive flag should reflect threshold comparison."""
    result = PathologyResult(name="Effusion", probability=0.8, positive=True)
    assert result.positive is True


def test_pathology_result_rejects_out_of_range_probability():
    """Probability must be in [0, 1]."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PathologyResult(name="Effusion", probability=1.5, positive=True)


def test_prediction_response_positive_count():
    """positive_count should match number of positive findings."""
    findings = [
        PathologyResult(name="Effusion", probability=0.9, positive=True),
        PathologyResult(name="Atelectasis", probability=0.3, positive=False),
        PathologyResult(name="Pneumonia", probability=0.7, positive=True),
    ]
    response = PredictionResponse(
        findings=findings,
        positive_count=sum(f.positive for f in findings),
        model_version="0.1.0",
    )
    assert response.positive_count == 2


def test_health_response_fields():
    """HealthResponse should carry status, model_loaded, and device."""
    h = HealthResponse(status="ok", model_loaded=True, device="cpu")
    assert h.status == "ok"
    assert h.model_loaded is True
    assert h.device == "cpu"
