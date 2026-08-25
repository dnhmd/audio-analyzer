import asyncio
import time
import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.audio_analyzer.core.audio import decode_audio, assess_quality
from src.audio_analyzer.core.inference import predict
from src.audio_analyzer.core.language import detect_language

log = structlog.get_logger()
router = APIRouter()

MIN_CHUNK_BYTES = 16000 * 2 * 1  # ~1s of 16kHz 16-bit mono

@router.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    await websocket.accept()
    log.info("ws_connected")

    buffer = bytearray()
    contact_id = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                import json
                data = json.loads(message["text"])
                contact_id = data.get("contact_id", "unknown")
                await websocket.send_json({"event": "session_started", "contact_id": contact_id})
                continue

            if "bytes" in message:
                chunk = message["bytes"]
                if not chunk:
                    break

                buffer.extend(chunk)

                if len(buffer) < MIN_CHUNK_BYTES:
                    await websocket.send_json({"event": "buffering", "buffered_bytes": len(buffer)})
                    continue

                start = time.monotonic()

                try:
                    audio_array = decode_audio(bytes(buffer))
                    quality = assess_quality(audio_array)

                    if quality == "insufficient":
                        await websocket.send_json({
                            "event": "prediction",
                            "audio_quality": quality,
                            "gender": {"prediction": "unknown", "confidence": 0.0},
                            "age_bracket": {"prediction": "unknown", "confidence": 0.0},
                            "language": None,
                            "processing_ms": round((time.monotonic() - start) * 1000, 2)
                        })
                        continue

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, predict, audio_array)

                    try:
                        language = await loop.run_in_executor(None, detect_language, audio_array)
                    except Exception:
                        language = None

                    processing_ms = round((time.monotonic() - start) * 1000, 2)

                    await websocket.send_json({
                        "event": "prediction",
                        "contact_id": contact_id,
                        "audio_quality": quality,
                        "gender": result["gender"],
                        "age_bracket": result["age_bracket"],
                        "language": language,
                        "processing_ms": processing_ms
                    })

                except Exception as e:
                    log.error("ws_inference_failed", error=str(e))
                    await websocket.send_json({"event": "error", "detail": str(e)})

    except WebSocketDisconnect:
        log.info("ws_disconnected")