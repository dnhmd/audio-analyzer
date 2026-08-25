# Audio Analyzer

A REST API for analyzing voice recordings and predicting **gender, age bracket, and language**.

The project is designed around logistics and voice AI use cases where incoming audio is not always clean. It supports common telephony codecs, noisy recordings, silence, low-quality signals, and degraded audio without requiring the caller to handle the underlying audio processing.

The service is built around a simple idea: take whatever audio comes in, normalize it, check whether it is usable, run the relevant models, and return the predictions in a consistent format.

## Requirements

You can run the service either with Docker or directly on your machine.

### Docker

- Docker
- Docker Compose

### Local development

- Python 3.12+
- `uv`
- `ffmpeg`
- `espeak` for the smoke test

On Ubuntu/Debian:

```bash
sudo apt install ffmpeg espeak -y
```

On macOS:

```bash
brew install ffmpeg espeak
```

## Quick Start

### Docker

Docker is the easiest way to run the complete service.

```bash
docker compose up
```

The model weights are around 1.3 GB and are downloaded automatically the first time the service starts. They are stored in a named Docker volume, so subsequent startups do not need an internet connection.

The service is ready once you see the model loaded message in the container logs:

```text
audio-analyzer | {"model": "audeering/wav2vec2-large-robust-24-ft-age-gender", "event": "model_loaded", ...}
```

### Local

For local development:

```bash
uv sync
make run
```

The API will be available on port `8000`.

## Smoke Test

Once the service is running, the quickest way to verify the complete pipeline is:

```bash
make smoke
```

The smoke test generates a small voice sample using `espeak` and sends it to the API.

You can also run it manually:

```bash
espeak "Hello my name is John and I am calling about my delivery" --stdout > /tmp/smoke.wav

curl -X POST http://localhost:8000/api/v1/analyze \
  -F "contact_id=smoke-001" \
  -F "audio=@/tmp/smoke.wav"
```

A successful request should return something similar to:

```json
{
  "contact_id": "smoke-001",
  "gender": {
    "prediction": "male",
    "confidence": 0.99
  },
  "age_bracket": {
    "prediction": "60+",
    "confidence": 0.68
  },
  "language": {
    "prediction": "en",
    "confidence": 0.98
  },
  "processing_ms": 2800,
  "audio_quality": "good"
}
```

The exact predictions and confidence values will vary depending on the generated audio and model behavior.

## API Reference

### `POST /api/v1/analyze`

Accepts a multipart audio upload and returns predictions for gender, age bracket, and language.

The API supports WAV, MP3, OGG, OPUS, G.711 µ-law/A-law, and any other format that ffmpeg can decode.

#### Request

- `contact_id`: An opaque correlation ID. It is returned in the response but is never logged or stored.
- `audio`: The audio file to analyze. Maximum size is 25 MB.

#### Response

```json
{
  "contact_id": "uuid",
  "gender": {
    "prediction": "male",
    "confidence": 0.99
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.64
  },
  "language": {
    "prediction": "en",
    "confidence": 0.98
  },
  "processing_ms": 3017,
  "audio_quality": "good"
}
```

`audio_quality` describes the quality of the input signal:

- `good`: The signal is reasonably clean and predictions should generally be more reliable.
- `degraded`: The recording contains significant noise, clipping, or a low SNR. Inference still runs, but confidence may be lower.
- `insufficient`: There is too much silence or the signal is too weak to make a useful prediction. Inference is skipped.

### `WebSocket /api/v1/ws/analyze`

There is also a WebSocket endpoint for streaming audio.

The client first sends a JSON message to initialize the session and then sends the audio as binary frames.

```python
import asyncio
import json
import websockets

async def stream():
    async with websockets.connect(
        "ws://localhost:8000/api/v1/ws/analyze"
    ) as ws:
        await ws.send(json.dumps({
            "contact_id": "test-001"
        }))

        with open("audio.wav", "rb") as f:
            await ws.send(f.read())

        print(await ws.recv())

asyncio.run(stream())
```

The streaming endpoint is intended for use cases where audio is available incrementally rather than as a complete file.

### `GET /health`

Returns the current model readiness status.

```json
{
  "status": "ok",
  "model": "loaded"
}
```

## Evaluation

The project includes a small evaluation harness for checking model performance against the Mozilla Common Voice dataset.

Download the dataset manually from the [Mozilla Common Voice datasets page](https://commonvoice.mozilla.org/en/datasets).

Then run:

```bash
PYTHONPATH=src uv run python tests/eval.py \
  --dataset-dir /path/to/cv-corpus/clips \
  --tsv /path/to/cv-corpus/validated.tsv \
  --limit 100
```

The `--limit` option is useful while developing if you only want to evaluate a subset of the dataset.

The harness reports:

- Accuracy
- Average prediction confidence
- Coverage

The complete results are also saved to:

```text
eval_results.json
```

This is primarily a sanity check and a way to compare changes to the inference pipeline. Common Voice is not a perfect representation of the target telephony workload, so its results should not be treated as a production benchmark.

## Design Decisions

### Audio Pipeline

The audio processing pipeline is intentionally simple:

```text
Raw audio
    ↓
ffmpeg decode
    ↓
16 kHz mono float32
    ↓
Signal quality checks
    ↓
┌─────────────────────────────┐
│ wav2vec2 → age + gender     │
│ Whisper  → language         │
└─────────────────────────────┘
    ↓
API response
```

ffmpeg handles codec conversion, including G.711 µ-law/A-law audio commonly found in VoIP systems.

The rest of the application works with a consistent 16 kHz mono float32 representation regardless of the original format.

### Age and Gender Model

The project uses `audeering/wav2vec2-large-robust-24-ft-age-gender`.

It is a wav2vec2-based model fine-tuned for age and gender prediction. I chose it because both predictions come from the same model and it was trained with robustness to noisy speech in mind, which makes it a reasonable starting point for telephony audio.

Other approaches considered:

- **SpeechBrain ECAPA-VOXCELEB**: Strong speaker embeddings, but would need a separate classifier for age and gender.
- **openSMILE**: Useful and interpretable acoustic features, but would require more manual feature engineering and tuning.
- **pyannote.audio**: Very capable for speaker diarization, but unnecessary for this use case.

### Language Detection

Language detection is handled by Whisper Tiny.

Whisper can identify the language of a speech sample as part of its normal processing, so there is no need to add another dedicated language classification model.

The `tiny` model keeps the additional latency relatively small while still providing useful language detection for reasonably clean speech.

### Quality Gate

The service performs a few basic checks before running inference.

**Silence**

If more than 80% of the recording is silent, the request is marked as `insufficient` and inference is skipped.

**Clipping**

If more than 1% of the samples are clipped, the recording is marked as `degraded`.

**SNR**

An estimated SNR below 10 dB also results in a `degraded` quality flag.

Degraded audio is still processed. The quality flag is there to make it clear that the prediction was made from a poor-quality signal.

The only case where inference is skipped is when the audio is considered `insufficient`.

### Privacy

Audio is deliberately kept ephemeral.

- Audio exists in memory only for the duration of the request.
- A temporary file may be created for ffmpeg decoding and is deleted immediately afterwards.
- `contact_id` is never logged or stored.
- Audio is not written to a database or object store.
- No audio is included in application telemetry.

The service is therefore stateless from an audio storage perspective.

## Concurrency and Scaling

The model is loaded once during application startup and treated as read-only.

Each request has its own processing state and there is no shared mutable request data.

The current implementation performs CPU inference directly, which means the inference work can block the async event loop. That is acceptable for a small deployment, but it would need to change for higher concurrency.

### Scaling to 1,000 Concurrent Calls

There are several obvious steps for scaling the service.

**1. Inference worker pool**

Move model inference out of the main request path and into a `ProcessPoolExecutor` sized according to the available CPU resources.

**2. Horizontal scaling**

The service is stateless, so multiple instances can run behind a load balancer without introducing shared application state.

**3. GPU inference**

For latency-sensitive workloads, GPU inference would be the biggest improvement.

The current CPU inference time is around ~2 seconds per chunk, which is well above the 500 ms target. A GPU-enabled deployment combined with small batches could significantly improve throughput.

**4. Queue-based processing**

At much larger volumes, synchronous HTTP requests may no longer be the right model.

A queue such as Kafka or SQS could sit between the API and inference workers:

```text
Client
   ↓
API
   ↓
Kafka / SQS
   ↓
Inference Workers
   ↓
Result
   ↓
Webhook / WebSocket
```

This would allow the API layer and inference layer to scale independently.

## Known Limitations

There are still a few areas that would need more work before treating this as a fully hardened production service.

- CPU inference is around ~2 seconds per chunk, which does not meet the 500 ms target.
- Age prediction is less reliable on synthetic or narrowband audio.
- The age and gender model was not specifically fine-tuned on telephony or VoIP recordings.
- Language detection can become less reliable with very short, noisy, or heavily degraded speech.
- The current SNR calculation is intentionally lightweight and should not be treated as a definitive measure of perceived audio quality.
- Prediction confidence is not calibrated against a production dataset.
- The API currently has no authentication.

The biggest improvement would probably be building a representative evaluation dataset from real telephony recordings and using it to validate the models and tune the quality thresholds. The current model being robust to noisy speech is useful, but that is not the same as being optimized for a specific VoIP environment.

## Development

For local development:

```bash
uv sync
make run
```

The project uses `uv` for dependency management and `make` for the common development commands.

### Tests

Run the test suite with:

```bash
make test
```

The smoke test can be run with:

```bash
make smoke
```

The evaluation harness is separate from the normal test suite because it requires an external dataset and is intended for model evaluation rather than application-level testing.