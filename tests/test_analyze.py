import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
from src.audio_analyzer.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_empty_audio():
    response = client.post(
        "/api/v1/analyze",
        data={"contact_id": "test-001"},
        files={"audio": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400

def test_analyze_invalid_audio():
    response = client.post(
        "/api/v1/analyze",
        data={"contact_id": "test-001"},
        files={"audio": ("test.wav", b"notaudiobytes", "audio/wav")}
    )
    assert response.status_code == 422

@patch("src.audio_analyzer.routers.analyze.decode_audio")
@patch("src.audio_analyzer.routers.analyze.assess_quality")
@patch("src.audio_analyzer.routers.analyze.predict")
def test_analyze_success(mock_predict, mock_quality, mock_decode):
    mock_decode.return_value = np.zeros(16000, dtype=np.float32)
    mock_quality.return_value = "good"
    mock_predict.return_value = {
        "gender": {"prediction": "male", "confidence": 0.95},
        "age_bracket": {"prediction": "31-45", "confidence": 0.72}
    }

    response = client.post(
        "/api/v1/analyze",
        data={"contact_id": "test-001"},
        files={"audio": ("test.wav", b"fakeaudio", "audio/wav")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gender"]["prediction"] == "male"
    assert body["age_bracket"]["prediction"] == "31-45"
    assert body["audio_quality"] == "good"
    assert "processing_ms" in body