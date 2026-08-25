import time
import structlog
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.audio_analyzer.core.audio import decode_audio, assess_quality
from src.audio_analyzer.core.inference import predict
from src.audio_analyzer.core.models import AnalyzeResponse, GenderResult, AgeBracketResult

log = structlog.get_logger()
router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    contact_id: str = Form(...),
    audio: UploadFile = File(...)
):
    start = time.monotonic()

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    try:
        audio_array = decode_audio(raw)
    except ValueError as e:
        log.error("decode_failed", error=str(e))
        raise HTTPException(status_code=422, detail="Audio decode failed")

    quality = assess_quality(audio_array)

    if quality == "insufficient":
        processing_ms = (time.monotonic() - start) * 1000
        return AnalyzeResponse(
            contact_id=contact_id,
            gender=GenderResult(prediction="unknown", confidence=0.0),
            age_bracket=AgeBracketResult(prediction="unknown", confidence=0.0),
            processing_ms=round(processing_ms, 2),
            audio_quality=quality
        )

    try:
        result = predict(audio_array)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Inference failed")

    processing_ms = (time.monotonic() - start) * 1000

    return AnalyzeResponse(
        contact_id=contact_id,
        gender=GenderResult(**result["gender"]),
        age_bracket=AgeBracketResult(**result["age_bracket"]),
        processing_ms=round(processing_ms, 2),
        audio_quality=quality
    )