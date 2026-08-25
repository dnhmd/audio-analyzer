import numpy as np
import whisper
import structlog

log = structlog.get_logger()

_model = None

def load_language_model():
    global _model
    if _model is None:
        log.info("loading_whisper", model="tiny")
        _model = whisper.load_model("tiny")
        log.info("whisper_loaded")

def detect_language(audio: np.ndarray) -> dict:
    load_language_model()

    audio_fp16 = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_fp16).to(_model.device)

    _, probs = _model.detect_language(mel)
    top_lang = max(probs, key=probs.get)
    confidence = round(probs[top_lang], 4)

    return {
        "prediction": top_lang,
        "confidence": confidence
    }