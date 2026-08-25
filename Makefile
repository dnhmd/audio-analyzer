.PHONY: run test docker lint

run:
	PYTHONPATH=src uv run uvicorn audio_analyzer.main:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=src uv run pytest tests/ -v

docker:
	docker compose up --build

smoke:
	espeak "Hello my name is John and I am calling about my delivery" --stdout > /tmp/smoke.wav
	curl -X POST http://localhost:8000/api/v1/analyze \
		-F "contact_id=smoke-001" \
		-F "audio=@/tmp/smoke.wav"