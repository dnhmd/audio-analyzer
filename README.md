# Audio Analyzer

A small production-focused REST API for analyzing voice recordings and predicting gender and age bracket.

The project is designed around logistics and voice AI use cases where the incoming audio is not always clean. It supports common telephony codecs, noisy recordings, silence, low-quality signals, and other real-world audio issues without making the API consumer deal with the underlying audio processing.

## Quick Start

```bash
docker compose up
```

The model weights are around 1.3 GB and are downloaded automatically from Hugging Face the first time the service starts. They are stored in a Docker named volume, so later startups can run without downloading the model again.

## API

### `POST /api/v1/analyze`

Accepts an audio file and returns predictions for gender and age bracket.

The API accepts WAV, MP3, OGG, OPUS, G.711 µ-law/A-law, and other formats supported by ffmpeg.

#### Request

- `contact_id`: An opaque correlation ID. It is used only for the response and is never logged or stored.
- `audio`: The audio file to analyze.

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
  "processing_ms": 3017,
  "audio_quality": "good"
}
```

`audio_quality` can be one of:

- `good`: The signal is reasonably clean and the predictions should be more reliable.
- `degraded`: The recording contains noticeable noise or has a low SNR. Inference still runs, but confidence may be lower.
- `insufficient`: There is too much silence or the signal is too weak to make a useful prediction. Inference is skipped.

### `GET /health`

Returns the current model readiness status.

## Design

### Model

The project uses [`audeering/wav2vec2-large-robust-24-ft-age-gender`](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender).

It is based on wav2vec2 and has been fine-tuned for age and gender prediction. I went with it mainly because it handles both predictions in a single model and was trained with robustness to noisy speech in mind, which makes it a better fit for the kind of audio expected from telephony systems.

A few alternatives I considered:

- **SpeechBrain ECAPA-VOXCELEB**: Very good for speaker embeddings, but would require an additional classifier for age and gender.
- **openSMILE**: A well-established approach for extracting acoustic features, but getting good predictions would require more manual feature engineering and domain-specific tuning.
- **pyannote.audio**: Excellent for speaker diarization, but more than what is needed for this particular service.

### Audio Pipeline

The processing pipeline is intentionally kept straightforward:

```text
Raw audio
   ↓
ffmpeg decode
   ↓
16 kHz mono float32
   ↓
Signal quality checks
   ↓
wav2vec2 inference
   ↓
API response
```

ffmpeg takes care of codec handling, including G.711 µ-law/A-law audio commonly found in VoIP systems.

Audio is not persisted. A short-lived temporary file is used during ffmpeg decoding and is removed immediately afterwards.

## Audio Quality Checks

Before running inference, the service performs a few basic signal checks.

### Silence

If more than 80% of the recording is silent, the request is marked as `insufficient` and inference is skipped.

There is little point in sending an essentially empty recording through a fairly expensive model.

### Clipping

If more than 1% of the samples are clipped, the recording is marked as `degraded`.

The prediction still runs, since clipped audio can still contain useful speech.

### SNR

An estimated SNR below 10 dB results in a `degraded` quality flag.

Again, inference is not skipped. The idea is to separate "we can probably still get something useful from this" from "there is not enough signal to bother running the model."

## Privacy

The service is intentionally stateless from an audio perspective.

- Audio stays in memory for the duration of the request.
- A temporary file may be created for ffmpeg decoding and is deleted immediately afterwards.
- `contact_id` is never logged or persisted.
- Audio is not stored in a database, filesystem, object store, or telemetry system.
- There is no request-level audio history.

The goal is to keep the service simple while minimizing the amount of sensitive data that exists outside the request lifecycle.

## Concurrency

The model is loaded once when the application starts and treated as read-only.

Each request has its own input and processing state, so there is no shared mutable request data.

The current implementation performs CPU inference directly, which means inference can block the async event loop. That's fine for a small deployment, but it would not be the approach I'd use for a high-throughput production system.

For higher concurrency, inference should be moved to a process pool, for example with `ProcessPoolExecutor`.

## Scaling to 1,000 Concurrent Calls

There are a few fairly straightforward ways to take this further.

### 1. Separate the API and inference workers

Move model inference out of the main async request path and into dedicated worker processes.

```text
                ┌──────────────┐
Requests ──────►│   API Layer  │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Worker Pool  │
                ├──────────────┤
                │ Model Worker │
                │ Model Worker │
                │ Model Worker │
                └──────────────┘
```

The number of workers can then be tuned based on available CPU and memory.

### 2. Horizontal Scaling

The service is stateless, so multiple instances can run behind a load balancer.

That makes scaling out fairly simple without having to introduce shared application state.

### 3. GPU Inference

The current CPU inference time is roughly 3 seconds per chunk.

For workloads where latency matters, running the model on a GPU would make a much bigger difference than trying to squeeze small optimizations out of the API layer.

Another option would be short-window batching. Incoming requests could be collected for a few milliseconds and processed together to make better use of the GPU.

### 4. Queue-Based Processing

For much larger workloads, I would decouple ingestion from inference completely.

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

This would make sense if the workload becomes large enough that synchronous HTTP requests are no longer the right abstraction.

## Known Limitations

There are still a few things I would change before calling this a fully hardened production service.

- CPU inference currently takes around 3 seconds per chunk, so it does not meet a 500 ms latency target.
- Age prediction is less reliable on synthetic or narrowband recordings.
- The model was not specifically fine-tuned on the target telephony/VoIP dataset.
- There is currently no authentication on the API endpoint.
- The SNR calculation is intentionally lightweight and should not be treated as a perfect measure of perceptual audio quality.
- Prediction confidence should not be interpreted as a calibrated probability without additional validation.

The biggest improvement would probably be domain-specific fine-tuning and evaluation on real telephony recordings. The model being robust to noisy speech is useful, but that is not the same thing as being optimized for a particular VoIP pipeline.

## Running Tests

```bash
uv run pytest tests/ -v
```

## Evaluation Harness

The project also includes a small evaluation harness for measuring model performance against the Mozilla Common Voice dataset.

The dataset needs to be downloaded manually from the [Mozilla Common Voice datasets page](https://commonvoice.mozilla.org/en/datasets).

Once the dataset is downloaded and extracted, run:

```bash
uv run python tests/eval.py \
  --dataset-dir /path/to/cv-corpus/clips \
  --tsv /path/to/cv-corpus/validated.tsv \
  --limit 100
```

The `--limit` argument can be used to run a smaller evaluation while developing. Removing it runs the evaluation across the available dataset.

The harness reports:

- **Accuracy**: How often the predicted gender matches the dataset label.
- **Average confidence**: The average confidence of the model's predictions.
- **Coverage**: The percentage of samples for which the model produces a usable prediction.

The complete results are also written to:

```text
eval_results.json
```

This is mainly intended as a lightweight way to sanity-check model performance and compare changes to the audio pipeline or inference setup.

The Common Voice evaluation should not be treated as a production benchmark. The target use case is telephony audio, while Common Voice contains a different distribution of recording conditions. A proper production evaluation would ideally use a representative set of real VoIP recordings with validated labels.

## Local Development

Install dependencies:

```bash
uv sync
```

Start the development server:

```bash
uv run uvicorn src.audio_analyzer.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

## Project Goal

The main goal of this project was not just to wrap a speech model behind a FastAPI endpoint.

The interesting part is everything around the model: accepting messy real-world audio, handling telephony codecs, checking whether the signal is worth processing, keeping audio ephemeral, and designing the service so the inference layer can eventually be scaled independently.

The current version keeps that architecture relatively small, while leaving a clear path towards GPU inference, worker-based processing, horizontal scaling, and eventually asynchronous queue-based processing.