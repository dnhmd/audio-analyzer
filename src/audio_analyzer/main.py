import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.audio_analyzer.routers.analyze import router
from src.audio_analyzer.core.inference import load_model
from src.audio_analyzer.core.language import load_language_model

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_language_model()
    yield

app = FastAPI(title="Audio Analyzer", version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}