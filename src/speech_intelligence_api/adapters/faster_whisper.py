"""Faster-Whisper large-v3 speech-recognition adapter."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

import anyio

from speech_intelligence_api.adapters.local_storage import LocalEphemeralBlobStore
from speech_intelligence_api.adapters.observability import NoopObservability
from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.enums import ChineseScript, LanguageCode
from speech_intelligence_api.domain.errors import (
    InvalidAudioError,
    ModelUnavailableError,
    ServiceError,
    UncertainLanguageError,
)
from speech_intelligence_api.domain.models import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from speech_intelligence_api.ports.observability import Observability

logger = logging.getLogger(__name__)


class _WhisperWord(Protocol):
    word: str
    start: float
    end: float
    probability: float


class _WhisperSegment(Protocol):
    text: str
    start: float
    end: float
    words: list[_WhisperWord] | None
    avg_logprob: float


class _WhisperInfo(Protocol):
    language: str
    language_probability: float
    all_language_probs: list[tuple[str, float]] | None


class _WhisperModel(Protocol):
    def transcribe(
        self,
        audio: str,
        **kwargs: Any,
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]:
        """Match the subset of Faster-Whisper used by this adapter."""


ModelFactory = Callable[[str], _WhisperModel]


class FasterWhisperSpeechRecognizer:
    """Lazily load one bounded Faster-Whisper model per worker process."""

    def __init__(
        self,
        settings: Settings,
        store: LocalEphemeralBlobStore,
        *,
        model_factory: ModelFactory | None = None,
        observability: Observability | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._model_factory = model_factory or self._build_model
        self._observability = observability or NoopObservability()
        self._models: dict[str, _WhisperModel] = {}
        self._model_lock = asyncio.Lock()
        self._inference_slots = asyncio.Semaphore(settings.asr_max_concurrency)
        self._supported_codes = frozenset(code.value for code in LanguageCode)

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Run source-language transcription with word timestamps and hotwords."""

        model = await self._get_model(self._model_name_for(request))
        with self._observability.span(
            "speech.inference.transcription",
            kind="internal",
            attributes={
                "speech.inference.operation": "transcription",
                "speech.inference.device": self._settings.asr_device,
            },
        ) as span:
            async with self._inference_slots:
                started_at = time.perf_counter()
                outcome = "failed"
                self._observability.inference_started(
                    operation="transcription",
                    device=self._settings.asr_device,
                )
                try:
                    result = await anyio.to_thread.run_sync(self._transcribe_sync, model, request)
                    outcome = "completed"
                    return result
                except ServiceError:
                    span.mark_error()
                    raise
                except Exception as exc:
                    span.mark_error()
                    logger.error(
                        "Speech-recognition inference failed",
                        extra={"exception_class": type(exc).__name__},
                    )
                    raise ModelUnavailableError from None
                finally:
                    span.set_attribute("speech.inference.outcome", outcome)
                    self._observability.inference_finished(
                        operation="transcription",
                        device=self._settings.asr_device,
                        outcome=outcome,
                        duration_seconds=time.perf_counter() - started_at,
                    )

    def _model_name_for(self, request: TranscriptionRequest) -> str:
        """Serve live partials from the smaller draft model when one is configured."""

        if request.draft and self._settings.asr_draft_model_name is not None:
            return self._settings.asr_draft_model_name
        return self._settings.asr_model_name

    async def _get_model(self, model_name: str) -> _WhisperModel:
        cached = self._models.get(model_name)
        if cached is not None:
            return cached
        async with self._model_lock:
            if model_name not in self._models:
                try:
                    self._models[model_name] = await anyio.to_thread.run_sync(
                        self._model_factory,
                        model_name,
                    )
                except Exception as exc:
                    logger.error(
                        "Speech-recognition model loading failed",
                        extra={"exception_class": type(exc).__name__},
                    )
                    raise ModelUnavailableError from None
        return self._models[model_name]

    def _build_model(self, model_name: str) -> _WhisperModel:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        download_root = (
            str(self._settings.asr_model_download_root)
            if self._settings.asr_model_download_root is not None
            else None
        )
        model = WhisperModel(
            model_name,
            device=self._settings.asr_device,
            compute_type=self._settings.asr_compute_type,
            cpu_threads=self._settings.asr_cpu_threads,
            num_workers=self._settings.asr_num_workers,
            download_root=download_root,
            local_files_only=self._settings.asr_model_local_files_only,
        )
        return cast(_WhisperModel, model)

    def _transcribe_sync(
        self,
        model: _WhisperModel,
        request: TranscriptionRequest,
    ) -> TranscriptionResult:
        audio_path = str(self._store.resolve_path(request.audio))
        if request.language.language is None:
            selected_language, language_confidence = self._detect_supported_language(
                model,
                audio_path,
            )
        else:
            selected_language = request.language.language
            language_confidence = 1.0

        raw_segments, _ = model.transcribe(
            audio_path,
            language=selected_language.value,
            task="transcribe",
            beam_size=1 if request.draft else self._settings.asr_beam_size,
            condition_on_previous_text=not request.draft,
            word_timestamps=request.word_timestamps and not request.draft,
            hotwords=", ".join(request.vocabulary) if request.vocabulary else None,
            vad_filter=True,
        )
        source_segments = list(raw_segments)
        segments = tuple(
            segment
            for source in source_segments
            if (segment := self._map_segment(source, selected_language)) is not None
        )
        if not segments:
            raise InvalidAudioError("No speech was detected in the uploaded audio.")

        chinese_script: ChineseScript | None = None
        if selected_language is LanguageCode.CHINESE:
            chinese_script = (
                request.language.chinese_script or self._settings.default_chinese_script
            )
        return TranscriptionResult(
            language=selected_language,
            language_confidence_estimate=language_confidence,
            text="".join(source.text for source in source_segments).strip(),
            segments=segments,
            chinese_script=chinese_script,
        )

    def _detect_supported_language(
        self,
        model: _WhisperModel,
        audio_path: str,
    ) -> tuple[LanguageCode, float]:
        _, info = model.transcribe(
            audio_path,
            language=None,
            task="transcribe",
            beam_size=1,
            word_timestamps=False,
            vad_filter=True,
            language_detection_segments=self._settings.asr_language_detection_segments,
            language_detection_threshold=1.0,
        )
        candidates_by_code: dict[str, float] = {}
        for code, probability in info.all_language_probs or ():
            if code in self._supported_codes:
                candidates_by_code[code] = max(candidates_by_code.get(code, 0.0), probability)
        if info.language in self._supported_codes:
            candidates_by_code[info.language] = max(
                candidates_by_code.get(info.language, 0.0),
                info.language_probability,
            )

        candidates = tuple(
            sorted(candidates_by_code.items(), key=lambda item: item[1], reverse=True)[:3]
        )
        if not candidates or candidates[0][1] < self._settings.language_confidence_threshold:
            raise UncertainLanguageError(
                self._settings.language_confidence_threshold,
                candidates,
            )
        return LanguageCode(candidates[0][0]), candidates[0][1]

    @staticmethod
    def _map_segment(
        source: _WhisperSegment,
        language: LanguageCode,
    ) -> TranscriptSegment | None:
        text = source.text.strip()
        if not text:
            return None
        start_seconds = max(0.0, float(source.start))
        end_seconds = max(start_seconds, float(source.end))
        words = tuple(
            mapped
            for word in source.words or ()
            if (mapped := FasterWhisperSpeechRecognizer._map_word(word, start_seconds, end_seconds))
            is not None
        )
        confidence = min(1.0, max(0.0, math.exp(float(source.avg_logprob))))
        return TranscriptSegment(
            text=text,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            language=language,
            words=words,
            confidence_estimate=confidence,
        )

    @staticmethod
    def _map_word(
        source: _WhisperWord,
        segment_start: float,
        segment_end: float,
    ) -> TranscriptWord | None:
        text = source.word.strip()
        if not text:
            return None
        start_seconds = min(segment_end, max(segment_start, float(source.start)))
        end_seconds = min(segment_end, max(start_seconds, float(source.end)))
        probability = min(1.0, max(0.0, float(source.probability)))
        return TranscriptWord(
            text=text,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            confidence_estimate=probability,
        )
