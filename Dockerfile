FROM python:3.12-slim

RUN apt-get update && apt-get install -y ffmpeg espeak curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install uv && uv sync --frozen

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "audio_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]