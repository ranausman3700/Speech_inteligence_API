"""Production adapter composition without transport-layer coupling."""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis

from speech_intelligence_api.adapters.celery_jobs import CeleryJobDispatcher
from speech_intelligence_api.adapters.faster_whisper import FasterWhisperSpeechRecognizer
from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.native_text import NativeTranscriptTextProcessor
from speech_intelligence_api.adapters.observability import NoopObservability
from speech_intelligence_api.adapters.pcm_wave import PcmWaveSnapshotStore
from speech_intelligence_api.adapters.pyannote_diarization import PyannoteSpeakerDiarizer
from speech_intelligence_api.adapters.pyav_audio import PyAvAudioPreprocessor
from speech_intelligence_api.adapters.redis_jobs import (
    RedisClientReadinessCheck,
    RedisJobStore,
    RedisReadinessCheck,
)
from speech_intelligence_api.adapters.redis_rate_limiting import (
    RedisRateLimitReadinessCheck,
    RedisSlidingWindowRateLimiter,
)
from speech_intelligence_api.adapters.silero_vad import SileroVoiceActivityDetector
from speech_intelligence_api.application.conversation_alignment import ConversationAssembler
from speech_intelligence_api.application.conversations import (
    ConversationJobWorker,
    ConversationSubmissionService,
)
from speech_intelligence_api.application.jobs import (
    JobManagementService,
    TranscriptionJobWorker,
)
from speech_intelligence_api.application.live_transcription import LiveTranscriptionService
from speech_intelligence_api.application.privacy import PrivacyCleanupService
from speech_intelligence_api.application.readiness import ReadinessCheck
from speech_intelligence_api.application.transcriptions import BatchTranscriptionService
from speech_intelligence_api.config import Settings
from speech_intelligence_api.ports.observability import Observability
from speech_intelligence_api.ports.rate_limiting import RateLimiter
from speech_intelligence_api.workers.celery_app import CONVERSATION_TASK_NAME, create_celery_app


@dataclass(slots=True)
class ApiRuntime:
    """Composed API services plus explicitly managed dependency resources."""

    transcription_service: BatchTranscriptionService
    conversation_service: ConversationSubmissionService | None = None
    live_transcription_service: LiveTranscriptionService | None = None
    job_service: JobManagementService | None = None
    rate_limiter: RateLimiter | None = None
    cleanup_service: PrivacyCleanupService | None = None
    readiness_checks: tuple[ReadinessCheck, ...] = ()
    redis_client: Redis | None = None
    broker_redis_client: Redis | None = None

    async def close(self) -> None:
        try:
            if self.redis_client is not None:
                await self.redis_client.aclose()
        finally:
            if self.broker_redis_client is not None:
                await self.broker_redis_client.aclose()


@dataclass(slots=True)
class WorkerRuntime:
    """Long-lived worker dependencies initialized once per worker process."""

    service: TranscriptionJobWorker
    job_store: RedisJobStore
    blob_store: LocalEphemeralBlobStore
    redis_client: Redis
    cleanup_service: PrivacyCleanupService


@dataclass(slots=True)
class ConversationWorkerRuntime:
    """Long-lived isolated diarization-worker dependencies."""

    service: ConversationJobWorker
    job_store: RedisJobStore
    blob_store: LocalEphemeralBlobStore
    redis_client: Redis


def build_api_runtime(
    settings: Settings,
    *,
    observability: Observability | None = None,
) -> ApiRuntime:
    """Compose direct processing and optional asynchronous job controls."""

    telemetry = observability or NoopObservability()
    blob_store = _blob_store(settings)
    preprocessor = PyAvAudioPreprocessor(blob_store)
    recognizer = FasterWhisperSpeechRecognizer(
        settings,
        blob_store,
        observability=telemetry,
    )
    text_processor = NativeTranscriptTextProcessor()
    live_service: LiveTranscriptionService | None = None
    conversation_service: ConversationSubmissionService | None = None
    job_store: RedisJobStore | None = None
    dispatcher: CeleryJobDispatcher | None = None
    job_service: JobManagementService | None = None
    readiness: tuple[ReadinessCheck, ...] = ()
    redis_client: Redis | None = None
    broker_redis_client: Redis | None = None
    rate_limiter: RedisSlidingWindowRateLimiter | None = None

    if settings.async_jobs_enabled or settings.rate_limit_enabled:
        redis_client = _redis_client(settings)

    if settings.async_jobs_enabled:
        if redis_client is None:
            raise RuntimeError("Redis is required for asynchronous jobs")
        job_store = RedisJobStore(redis_client, key_prefix=settings.job_key_prefix)
        dispatcher = CeleryJobDispatcher(
            create_celery_app(settings),
            observability=telemetry,
            task_names_by_queue={
                "diarization": CONVERSATION_TASK_NAME,
            },
        )
        job_service = JobManagementService(
            job_store=job_store,
            dispatcher=dispatcher,
            blob_store=blob_store,
        )
        broker_redis_client = _broker_redis_client(settings)
        readiness = (
            RedisReadinessCheck(job_store),
            RedisClientReadinessCheck(broker_redis_client, name="redis_broker"),
        )
        if settings.diarization_enabled:
            conversation_service = ConversationSubmissionService(
                store=blob_store,
                preprocessor=preprocessor,
                job_store=job_store,
                job_dispatcher=dispatcher,
                max_upload_bytes=settings.max_upload_bytes,
                max_audio_duration_seconds=settings.max_audio_duration_seconds,
                privacy_ttl_seconds=settings.private_artifact_ttl_seconds,
                max_pending_jobs=settings.max_pending_jobs,
                max_expected_speakers=settings.diarization_max_expected_speakers,
                speaker_confidence_threshold=settings.speaker_confidence_threshold,
            )

    if settings.rate_limit_enabled:
        if redis_client is None:
            raise RuntimeError("Redis is required for distributed rate limiting")
        rate_limiter = RedisSlidingWindowRateLimiter(
            redis_client,
            key_prefix=settings.job_key_prefix,
        )
        if not readiness:
            readiness = (RedisRateLimitReadinessCheck(rate_limiter),)

    cleanup_service = PrivacyCleanupService(
        blob_store=blob_store,
        job_store=job_store,
    )

    service = BatchTranscriptionService(
        store=blob_store,
        preprocessor=preprocessor,
        recognizer=recognizer,
        text_processor=text_processor,
        max_upload_bytes=settings.max_upload_bytes,
        max_audio_duration_seconds=settings.max_audio_duration_seconds,
        sync_max_audio_duration_seconds=settings.sync_max_audio_duration_seconds,
        privacy_ttl_seconds=settings.private_artifact_ttl_seconds,
        job_store=job_store,
        job_dispatcher=dispatcher,
        max_pending_jobs=settings.max_pending_jobs,
        long_audio_queue_threshold_seconds=settings.long_audio_queue_threshold_seconds,
    )
    if settings.live_transcription_enabled:
        live_service = LiveTranscriptionService(
            detector=SileroVoiceActivityDetector(
                threshold=settings.live_vad_threshold,
                min_speech_ms=settings.live_vad_min_speech_ms,
                min_silence_ms=settings.live_vad_min_silence_ms,
                speech_pad_ms=settings.live_speech_pad_ms,
            ),
            snapshot_store=PcmWaveSnapshotStore(blob_store),
            recognizer=recognizer,
            text_processor=text_processor,
            max_sessions=settings.live_max_sessions,
            max_session_seconds=settings.live_max_session_seconds,
            max_utterance_seconds=settings.live_max_utterance_seconds,
            max_chunk_bytes=settings.live_max_chunk_bytes,
            end_silence_ms=settings.live_end_silence_ms,
            analysis_window_seconds=settings.live_vad_analysis_window_seconds,
            pre_roll_ms=settings.live_pre_roll_ms,
            partial_min_audio_seconds=settings.live_partial_min_audio_seconds,
            partial_interval_seconds=settings.live_partial_interval_seconds,
            privacy_ttl_seconds=settings.private_artifact_ttl_seconds,
        )
    return ApiRuntime(
        transcription_service=service,
        conversation_service=conversation_service,
        live_transcription_service=live_service,
        job_service=job_service,
        rate_limiter=rate_limiter,
        cleanup_service=cleanup_service,
        readiness_checks=readiness,
        redis_client=redis_client,
        broker_redis_client=broker_redis_client,
    )


def build_transcription_service(settings: Settings) -> BatchTranscriptionService:
    """Compose the transcription use case for compatibility and focused tests."""

    return build_api_runtime(settings).transcription_service


def build_worker_runtime(
    settings: Settings,
    *,
    observability: Observability | None = None,
) -> WorkerRuntime:
    """Compose a worker process with shared storage and Redis job state."""

    telemetry = observability or NoopObservability()
    blob_store = _blob_store(settings)
    redis_client = _redis_client(settings)
    job_store = RedisJobStore(redis_client, key_prefix=settings.job_key_prefix)
    service = TranscriptionJobWorker(
        job_store=job_store,
        blob_store=blob_store,
        recognizer=FasterWhisperSpeechRecognizer(
            settings,
            blob_store,
            observability=telemetry,
        ),
        text_processor=NativeTranscriptTextProcessor(),
    )
    cleanup_service = PrivacyCleanupService(blob_store=blob_store, job_store=job_store)
    return WorkerRuntime(
        service=service,
        job_store=job_store,
        blob_store=blob_store,
        redis_client=redis_client,
        cleanup_service=cleanup_service,
    )


def build_conversation_worker_runtime(
    settings: Settings,
    *,
    observability: Observability | None = None,
) -> ConversationWorkerRuntime:
    """Compose the dedicated pyannote/ASR conversation worker."""

    if not settings.diarization_enabled:
        raise RuntimeError("speaker diarization is disabled")
    telemetry = observability or NoopObservability()
    blob_store = _blob_store(settings)
    redis_client = _redis_client(settings)
    job_store = RedisJobStore(redis_client, key_prefix=settings.job_key_prefix)
    service = ConversationJobWorker(
        job_store=job_store,
        blob_store=blob_store,
        recognizer=FasterWhisperSpeechRecognizer(
            settings,
            blob_store,
            observability=telemetry,
        ),
        text_processor=NativeTranscriptTextProcessor(),
        diarizer=PyannoteSpeakerDiarizer(
            settings,
            blob_store,
            observability=telemetry,
        ),
        assembler=ConversationAssembler(),
    )
    return ConversationWorkerRuntime(
        service=service,
        job_store=job_store,
        blob_store=blob_store,
        redis_client=redis_client,
    )


def _blob_store(settings: Settings) -> LocalEphemeralBlobStore:
    return LocalEphemeralBlobStore(
        settings.temp_storage_root,
        read_chunk_bytes=settings.upload_chunk_bytes,
    )


def _redis_client(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_job_url.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        health_check_interval=30,
        retry_on_timeout=True,
    )


def _broker_redis_client(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.celery_broker_url.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        health_check_interval=30,
        retry_on_timeout=True,
    )
