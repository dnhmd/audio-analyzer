import asyncio
import tempfile
import os
import io
import subprocess
import numpy as np
import structlog
from src.audio_analyzer.config import settings

log = structlog.get_logger()

def decode_audio(raw_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", tmp_path,
            "-ar", str(settings.sample_rate),
            "-ac", "1",
            "-f", "f32le",
            "pipe:1"
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=settings.inference_timeout_sec
        )
        if result.returncode != 0:
            raise ValueError(f"ffmpeg decode failed: {result.stderr.decode()}")
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio
    finally:
        os.unlink(tmp_path)

def assess_quality(audio: np.ndarray) -> str:
    """SNR estimation, silence ratio, clipping detection."""
    if len(audio) == 0:
        return "insufficient"

    silence_ratio = np.mean(np.abs(audio) < 0.01)
    if silence_ratio > settings.silence_ratio_threshold:
        return "insufficient"

    clipping_ratio = np.mean(np.abs(audio) > 0.99)
    if clipping_ratio > 0.01:
        return "degraded"

    signal_power = np.mean(audio ** 2)
    noise_floor = np.percentile(np.abs(audio), 10) ** 2
    if noise_floor == 0:
        return "degraded"
    
    snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10))
    if snr_db < settings.snr_threshold_db:
        return "degraded"

    return "good"