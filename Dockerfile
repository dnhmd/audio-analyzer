FROM python:3.12-alpine

RUN apk add --no-cache ffmpeg espeak curl

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN pip install uv && uv sync --frozen

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "audio_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]