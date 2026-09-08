# Speech Intelligence API

Production-oriented FastAPI backend for multilingual native-script transcription,
real-time dictation and speaker diarization.

The repository currently contains the Phase 8 backend: secure batch transcription,
Redis-backed asynchronous jobs, authenticated WebSocket dictation with Silero VAD, and
asynchronous recorded-conversation transcription with pyannote speaker diarization,
overlap/uncertainty metadata, raw plus speaker-formatted native-script output, and
distributed caller rate limiting, privacy-safe Prometheus metrics, and W3C/OTLP tracing.
Every speech endpoint has a CPU execution path; CUDA is an optional latency accelerator.

## Requirements

- Python 3.11
- Docker 20.10 or newer (optional)
- Redis 8 when asynchronous jobs run outside Docker Compose
- Enough memory and disk space for the selected Faster-Whisper model
- A CUDA-capable runtime is optional; CPU inference is supported
- A Hugging Face account/token and accepted pyannote Community-1 model terms are required
  only when Phase 5 diarization is enabled

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy src tests
```

Copy `.env.example` to `.env`, replace all placeholders, and calculate every configured
API-key digest as:

```text
hex(HMAC-SHA256(api_key_hmac_secret, raw_api_key))
```

The raw API key and HMAC secret belong in a secret manager or deployment environment,
never in the repository.

Run the service:

```powershell
.\.venv\Scripts\python -m uvicorn speech_intelligence_api.entrypoints.http.app:create_app --factory
```

Liveness is available at `/health/live`, readiness at `/health/ready`, and the
authenticated capability contract at `/v1/capabilities`.

## Production HTTP boundary

Terminate TLS at a hardened reverse proxy or managed load balancer and forward traffic
to the Compose API port. Compose binds that plaintext port to `127.0.0.1` by default, so
it is not exposed on every host interface. Set `SPEECH_API_BIND_ADDRESS` only when the
backend is on a controlled private network. Configure `SPEECH_API_FORWARDED_ALLOW_IPS`
with the exact proxy source IP; never trust all forwarded headers on a publicly reachable
backend.

Staging and production fail startup unless authentication, distributed rate limiting,
security headers, HSTS, and at least one non-local trusted hostname are configured.
`SPEECH_API_TRUSTED_HOSTS` accepts exact hostnames only. Production CORS origins must use
HTTPS. HTTPS responses include HSTS, and every HTTP response carries anti-caching,
anti-sniffing, frame, referrer, and browser-permission headers. This protects private
transcripts from intermediary/browser storage without changing the REST payloads.

The Compose services run read-only as non-root users, drop all Linux capabilities, use a
bounded PID budget, enable an init process, and keep both Redis services off host ports.
Place API/Hugging Face secrets in the deployment secret manager; `.env.example` contains
configuration shapes only.

Before a release, validate the exact deployment environment without printing its values,
then run the same quality gates as CI:

```powershell
.\.venv\Scripts\python -c "from speech_intelligence_api.config import Settings; Settings()"
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m mypy src tests
.\.venv\Scripts\python -m pytest
docker compose config --quiet
```

After TLS routing is active, check `/health/live`, `/health/ready`, one authenticated
`/v1/capabilities` request, and a short non-sensitive transcription fixture. Do not use
production recordings for deployment smoke tests. A readiness failure must remove the
instance from traffic; a liveness failure may restart it. Rotate API and model-provider
credentials through the secret manager without committing them or logging raw values.

## Distributed rate limiting

Phase 6A uses the Redis job-state database for exact sliding-window admission across all
API processes. Authenticated HTTP and WebSocket callers are isolated by a non-reversible
API-key digest identifier. When authentication is intentionally disabled for local
development, the client IP is used instead. Raw API keys and IP addresses are never
written into Redis keys.

General HTTP requests, upload submissions, and live-session starts have independent
limits configured by `SPEECH_API_RATE_LIMIT_HTTP_REQUESTS`,
`SPEECH_API_RATE_LIMIT_UPLOAD_REQUESTS`, and
`SPEECH_API_RATE_LIMIT_LIVE_SESSIONS`. They share
`SPEECH_API_RATE_LIMIT_WINDOW_SECONDS`. Successful HTTP responses include bounded
`X-RateLimit-*` metadata, including an explicit reset-after duration. Rejected HTTP
requests return `429` with `Retry-After`; rejected WebSocket starts emit a fatal
`rate_limited` event and close with code `4429`.

Rate limiting is mandatory in staging and production. If Redis is unavailable, admission
fails closed with a sanitized dependency error and readiness reports Redis unavailable.
Health endpoints remain public and are not part of authenticated caller quotas.

## Metrics and tracing

Set `SPEECH_API_METRICS_ENABLED=true` to expose `/metrics`. When API-key authentication
is enabled, the scrape must provide the same `X-API-Key` header as other protected
operations. Metrics scrapes do not consume public API rate-limit capacity. The endpoint
is intentionally omitted from OpenAPI and returns `Cache-Control: no-store`.

Prometheus measurements use only bounded operational labels: HTTP method, route template,
status code, configured queue, job kind, inference operation/device, and outcome. Raw URL
paths, job IDs, API keys, IP addresses, filenames, audio, vocabulary, language text, and
transcripts are never accepted as metric labels. Available series cover HTTP throughput
and latency, job submissions and queue wait, worker duration, and ASR/diarization duration
and in-flight work.

Tracing is opt-in. Set `SPEECH_API_TRACING_ENABLED=true` only when the configured
`SPEECH_API_OTLP_TRACES_ENDPOINT` is reachable. The API accepts the W3C `traceparent`
header, creates route-template server spans, injects trace context into Celery message
headers, and continues it in worker and model-inference spans. Unbounded baggage and
`tracestate` values are deliberately not propagated. Trace
attributes follow the same metadata-only policy, and exception messages are not exported.
`SPEECH_API_TRACE_SAMPLE_RATIO` controls root-span sampling from `0.0` to `1.0` while
honoring an upstream sampling decision.

For example, with authentication enabled:

```powershell
curl.exe -H "X-API-Key: replace-with-your-key" http://127.0.0.1:8000/metrics
```

## Privacy lifecycle and automatic deletion

Private state has two independent deletion boundaries. Redis job payloads, idempotency
records, transcripts, formatted conversation transcripts, and results use one absolute
expiry that cannot be extended by progress updates or result writes. Docker Compose runs
this private state in the memory-only `redis-state` service with snapshots and AOF
disabled. The separately persisted `redis-broker` contains only opaque job IDs and
bounded tracing/timing headers; audio, vocabulary, transcripts, and results are never
published to Celery.

Raw uploads, normalized audio, live inference snapshots, and diarization inputs use
random private filenames and an earlier filesystem expiry. The default 30-second cleanup
interval is subtracted from the 30-minute privacy limit, leaving a scheduled-deletion
window without extending retention. Successful, failed, cancelled, expired, and replayed
work all use idempotent deletion. The API performs fail-closed expired-file cleanup before
serving after a restart, while Celery Beat and the isolated cleanup worker continue it
during normal operation with bounded retries. If the local stack is powered off, no
process can run; overdue files are purged before the restarted API accepts traffic.

The `audio-data` volume is required only to share temporary normalized audio between API
and worker containers. Do not back it up or mount it into a public path. Production object
storage must enforce an equivalent or shorter provider-side lifecycle policy. Reducing
`SPEECH_API_PRIVACY_TTL_SECONDS` is supported; increasing it beyond 1,800 seconds is
rejected.

Upgrades from an earlier single-Redis Compose stack may leave its now-unused
`speech-intelligence_redis-data` volume on the host. Treat that volume as sensitive and
retire it through an approved destructive-change procedure; Compose intentionally never
deletes an existing operator volume automatically.

## Load and resilience validation

Phase 6C includes a backend-only client for repeatable asynchronous job-admission tests.
It sends a small supported audio fixture, creates a unique idempotency key per request,
and reports only aggregate status, throughput, and p50/p95/p99 latency. It never prints
the API key, uploaded path, response body, audio, or transcript.

Run the test only against an isolated environment. Install the optional client, put the
key in the current terminal, and submit the 1,000-request admission contract:

```powershell
.\.venv\Scripts\python -m pip install -e ".[load]"
$env:SPEECH_LOAD_TEST_API_KEY = "replace-with-an-isolated-test-key"
.\.venv\Scripts\speech-load-test --audio C:\path\to\short.wav --submissions 1000 --concurrency 100
Remove-Item Env:SPEECH_LOAD_TEST_API_KEY
```

The isolated deployment must set `SPEECH_API_MAX_PENDING_JOBS` to at least the target
submission count. Its upload and general HTTP rate limits must also admit that count in
the configured window, or rate limiting should be intentionally disabled only for the
isolated test. Keep the audio fixture short: admitted jobs are real jobs when workers are
running and therefore consume inference capacity. Stop workers when validating admission
only, then delete/expire the resulting ephemeral jobs before normal traffic resumes.

By default, only HTTP `202` passes. Add `--allow-status 503` when deliberately verifying
queue backpressure. A full queue returns a sanitized `capacity_exceeded` problem with
`Retry-After`; callers should delay before retrying. Docker Compose also bounds active
HTTP work, maintains a larger socket backlog, and gives in-flight requests a graceful
shutdown window. Automated tests exercise 1,000 concurrent in-process HTTP admissions
without invoking the ASR model, plus Redis's atomic 1,000-job capacity boundary.

## Run the backend stack

Copy `.env.example` to `.env`, replace the authentication placeholders, then run:

```powershell
docker compose up --build
```

This starts the API, isolated private-state and broker Redis services, short/long
transcription workers, the cleanup worker, and Celery Beat. Neither Redis role is
published to the host. Uploaded audio is shared
only through a private Docker volume and cleanup retries remove artifacts left by a
crashed process. CPU workers use the multilingual `medium` model by default so the
stack remains usable with an approximately 8 GB Docker memory limit. CPU inference
uses CTranslate2 `int8`; the GPU profile overrides this with CUDA `float16`. Set
`SPEECH_API_ASR_MODEL_NAME` in `.env` only when the CPU host has enough memory for a
larger model. Scale consumers independently:

```powershell
docker compose up --scale worker-short=3 --scale worker-long=2
```

For an NVIDIA GPU host with NVIDIA Container Toolkit installed, use the CUDA 12/cuDNN
batch worker and do not start the CPU transcription workers. The batch GPU profile
explicitly uses multilingual `large-v3`, independently of the CPU model setting:

```powershell
docker compose --profile gpu up --build redis-state redis-broker api worker-gpu worker-cleanup beat
```

Live transcription runs inside the API process so it can retain a warm model and avoid
Celery round trips. To allocate an NVIDIA GPU to live dictation, stop the CPU API and
start the dedicated GPU API while leaving batch jobs on CPU:

```powershell
docker compose stop api
docker compose --profile live-gpu up --build redis-state redis-broker api-gpu worker-short worker-long worker-cleanup beat
```

Running `api-gpu` and `worker-gpu` together loads one model in each process and therefore
requires enough VRAM for both. With a single 8 GB GPU, dedicate it to live dictation and
keep the batch workers on CPU.

## Transcript and summary export

`POST /v1/exports` turns a transcript the caller already holds into a downloadable
document, so one endpoint serves synchronous transcriptions, asynchronous job results and
live dictation alike:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/exports" `
  -H "X-API-Key: replace-with-your-api-key" `
  -H "Content-Type: application/json" `
  -d '{\"transcript\":\"...\",\"content\":\"summary\",\"format\":\"pdf\"}' `
  --output summary.pdf
```

`content` selects `transcript` (every sentence) or `summary` (the sentences carrying the
transcript's most distinctive vocabulary, as bullets). `format` selects `txt` or `pdf`, and
an optional `title` becomes both the document heading and the download name. The response
is the file itself with `Content-Disposition: attachment` and `Cache-Control: no-store`.

Summaries are extractive, so export costs no inference and no network call: sentences are
ranked by term frequency after discarding the most common quarter of the vocabulary, which
approximates stopword removal without a per-language word list, and scripts that do not
separate words with spaces are scored on character bigrams. A transcript under four
sentences returns its sentences unchanged, because such a transcript is already its own
summary. Measured on this project, a plain-text export takes under 10 ms and a PDF about
30-150 ms.

Nothing is stored to serve a download, so exports are unaffected by the privacy TTL, and a
caller can export a result it already fetched even after the job expired.

### PDF export fonts

Plain-text export always works. PDF must embed a font that contains the glyphs it draws,
and Arabic, Devanagari, Bengali and Thai additionally need shaping so letters join and
stack correctly.

The Docker image already installs `.[export]` and downloads the fonts below into `/fonts`,
which Compose passes as `SPEECH_API_EXPORT_FONT_ROOT`, so PDF works out of the box for
every supported language except Chinese, Japanese and Korean:

```text
NotoSans-Regular.ttf            Latin, Cyrillic, Greek
NotoSansArabic-Regular.ttf      Arabic
NotoSansHebrew-Regular.ttf      Hebrew
NotoSansDevanagari-Regular.ttf  Hindi
NotoSansBengali-Regular.ttf     Bengali
NotoSansThai-Regular.ttf        Thai
NotoSansCJK-Regular.ttf         Chinese, Japanese, Korean (not bundled)
```

The CJK face is excluded on purpose: it alone is roughly twenty times the size of that
whole set. Add it to the image, or mount it into the font directory, only where callers
need those languages.

Outside Docker, install the renderer with `pip install -e ".[export]"` and point
`SPEECH_API_EXPORT_FONT_ROOT` at a directory holding the same filenames, downloaded from
the [Noto releases](https://github.com/notofonts/notofonts.github.io).

When the text needs a font that is not installed, the request fails with a sanitized `422`
naming the missing file and pointing at `txt`, rather than returning a document with
missing glyphs. The same happens when the renderer itself is unavailable, so a deployment
without `.[export]` still serves plain text.

## Recorded multi-speaker conversations

`POST /v1/conversations` always returns HTTP `202`; ASR and speaker diarization run on a
dedicated `diarization` Celery queue. The completed result contains word timestamps,
speaker-labelled segments, speaker-confidence estimates, overlap and uncertain-assignment
flags, the untouched native-script transcript, and a readable speaker-formatted transcript.
Set `expected_speakers` only when the count is known (2-20 by default).

The default model is `pyannote/speaker-diarization-community-1`. Before starting its
worker, accept the model conditions on Hugging Face and inject a read-only token as
`SPEECH_API_DIARIZATION_HUGGINGFACE_TOKEN`. The model is CC-BY-4.0 licensed. Token and
audio data must never be committed. Model telemetry is disabled by the adapter and the
container. To enable the API and start the isolated GPU worker:

```powershell
# In .env: SPEECH_API_DIARIZATION_ENABLED=true
# In .env: SPEECH_API_DIARIZATION_HUGGINGFACE_TOKEN=<read-only deployment secret>
docker compose --profile diarization-gpu up --build redis-state redis-broker api worker-short worker-long worker-diarization worker-cleanup beat
```

Submit a recording and then poll the returned links:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/conversations" `
  -H "X-API-Key: replace-with-your-api-key" `
  -H "Idempotency-Key: conversation-retry-key-0001" `
  -F "file=@C:\path\to\meeting.m4a;type=audio/mp4" `
  -F "language_mode=explicit" `
  -F "language=en" `
  -F "expected_speakers=2"
```

For local CPU verification, install `.[diarization]`, set diarization enabled with both
ASR and diarization devices set to `cpu`, and run a Celery worker on `--queues
diarization`. CPU is contract-compatible but usually too slow for production recordings;
the dedicated GPU profile is the intended deployment. All normalized audio, job state,
and results retain the existing absolute 30-minute maximum privacy TTL and are deleted on
completion, cancellation, failure, or expiry cleanup.

For a fully containerized CPU-only conversation stack, accept the gated model terms and
set these deployment secrets/options in `.env`:

```text.
SPEECH_API_DIARIZATION_ENABLED=true
SPEECH_API_DIARIZATION_HUGGINGFACE_TOKEN=<read-only deployment secret>
SPEECH_API_ASR_MODEL_NAME=medium
```

Then start the isolated CPU worker profile:

```powershell
docker compose --profile diarization-cpu up --build redis-state redis-broker api worker-short worker-long worker-diarization-cpu worker-cleanup beat
```

This profile runs Faster-Whisper with CPU `int8` and pyannote on CPU. It implements the
same conversation contract and native-script output as the GPU profile, but recorded
conversations can take several times their audio duration. Do not run the CPU and GPU
diarization workers on the same queue unless duplicate capacity is intentional.

## Live-transcription WebSocket protocol

Connect to `ws://127.0.0.1:8000/v1/live-transcription` locally or the equivalent `wss://`
production URL. Supply the API key in the `X-API-Key` handshake header. Browser clients
that cannot set custom WebSocket headers may place it only in the first JSON message;
the value is authenticated and never echoed or persisted.

The first message must be JSON:

```json
{
  "type": "start",
  "api_key": "browser-only-api-key",
  "encoding": "pcm_s16le",
  "sample_rate_hz": 16000,
  "language_mode": "explicit",
  "language": "en",
  "vocabulary": ["FastAPI"],
  "word_timestamps": true
}
```

After the `ready` event, send binary frames containing mono 16 kHz little-endian signed
16-bit PCM. Each frame must stay within `max_chunk_bytes` reported by `ready`. The server
emits `speech_started`, best-effort `partial`, and accurate `final` events. JSON controls
are `{"type":"commit"}`, `{"type":"ping"}`, and `{"type":"stop"}`. Invalid input,
capacity pressure, uncertain automatic language, and dependency failures use sanitized
`error` events. Streamed PCM remains only in bounded session memory; private WAV
snapshots exist only during an inference call and are deleted in `finally` cleanup.

Partial events are replaceable drafts, so they decode greedily and skip the word-timestamp
alignment pass that only a final result needs. Setting `SPEECH_API_ASR_DRAFT_MODEL_NAME` to a
smaller model such as `tiny` serves partials from it while finals keep using
`SPEECH_API_ASR_MODEL_NAME`, which lowers dictation latency without changing the accuracy of
the text a caller keeps. Both models stay loaded in the API process, so budget memory for the
pair.

CPU inference cannot guarantee interactive partial latency. Whisper decodes buffered windows
rather than streaming tokens, so partials arrive in chunks instead of word by word. For
within-seconds dictation, use the `live-gpu` profile and keep the model warm.

### Test live dictation from your microphone

Install the optional terminal-client dependency once:

```powershell
.\.venv\Scripts\python -m pip install -e ".[microphone]"
```

List available input devices when the Windows default microphone is not the intended one:

```powershell
.\.venv\Scripts\speech-mic-test --list-devices
```

Start a 30-second English session using the default microphone:

```powershell
.\.venv\Scripts\speech-mic-test --language en --duration 30
```

The terminal prints `[partial]` text while speech is in progress and `[final]` text after
an utterance boundary or at the end of the session. Pass `--device 2` to select a device
index. The client prefers an input device with a native 16 kHz mode when `--device` is
omitted. When API authentication is enabled, set the key only in the current terminal:

```powershell
$env:SPEECH_API_CLIENT_API_KEY = "replace-with-your-local-test-key"
.\.venv\Scripts\speech-mic-test --language en --duration 30
```

The client does not record audio to disk and never prints the API key.
It waits up to 600 seconds for the final result by default because CPU inference can be
slow; override this with `--final-timeout SECONDS`. A warm GPU API normally finishes much
sooner.

## Test a transcription locally

For a local-only smoke test, start PowerShell in the repository and set non-secret
development overrides:

```powershell
$env:SPEECH_API_ENVIRONMENT = "local"
$env:SPEECH_API_AUTH_ENABLED = "false"
$env:SPEECH_API_ASR_DEVICE = "cpu"
$env:SPEECH_API_ASR_COMPUTE_TYPE = "int8"
.\.venv\Scripts\python -m uvicorn speech_intelligence_api.entrypoints.http.app:create_app --factory
```

The first transcription can take longer while Faster-Whisper downloads the configured
model.
In another terminal, submit a WAV, MP3, FLAC, OGG/Opus, M4A, WebM, or AAC file:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/transcriptions" `
  -F "file=@C:\path\to\speech.wav;type=audio/wav" `
  -F "language_mode=explicit" `
  -F "language=en" `
  -F "processing_mode=sync" `
  -F "word_timestamps=true"
```

For automatic selection, omit `language` and send
`-F "language_mode=automatic"`. For Chinese, also send
`-F "chinese_script=traditional"` or `simplified`. The API always uses transcription
mode and returns the source language; it does not translate or Romanize output.

With asynchronous jobs enabled, explicitly queue a recording:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/transcriptions" `
  -H "X-API-Key: replace-with-your-api-key" `
  -H "Idempotency-Key: retry-safe-random-key-0001" `
  -F "file=@C:\path\to\speech.wav;type=audio/wav" `
  -F "language_mode=explicit" `
  -F "language=en" `
  -F "processing_mode=async"
```

The API returns HTTP `202` and links for:

```text
GET    /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/result
POST   /v1/jobs/{job_id}/cancel
DELETE /v1/jobs/{job_id}
```

In `auto` mode, recordings above `SPEECH_API_SYNC_MAX_AUDIO_DURATION_SECONDS` are queued.
In `sync` mode they are rejected rather than silently occupying an HTTP worker. Job
state, results, idempotency records, normalized audio, and failure metadata never outlive
the configured privacy TTL. Broker messages contain only opaque job IDs.

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs` when documentation is enabled.
