import structlog
from fastapi import FastAPI
from src.audio_analyzer.routers.analyze import router

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

app = FastAPI(title="Audio Analyzer", version="0.1.0")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}