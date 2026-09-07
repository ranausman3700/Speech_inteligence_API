"""Bounded real-time dictation orchestration independent of WebSocket transport."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from speech_intelligence_api.domain.enums import LiveEventType
from speech_intelligence_api.domain.errors import (
    InvalidAudioError,
    LiveCapacityExceededError,
    LiveSessionLimitError,
    UncertainLanguageError,
)
from speech_intelligence_api.domain.models import (
    LiveSessionEvent,
    LiveTranscriptionOptions,
    TranscriptionRequest,
    TranscriptionResult,
)
from speech_intelligence_api.ports.asr import SpeechRecognizer
from speech_intelligence_api.ports.live_audio import PcmAudioSnapshotStore
from speech_intelligence_api.ports.text import TranscriptTextProcessor
from speech_intelligence_api.ports.vad import VoiceActivityDetector

_BYTES_PER_SAMPLE = 2
_VAD_WINDOW_SAMPLES = 512


class LiveTranscriptionService:
    """Admit a bounded number of independent in-process live sessions."""

    def __init__(
        self,
        *,
        detector: VoiceActivityDetector,
        snapshot_store: PcmAudioSnapshotStore,
        recognizer: SpeechRecognizer,
        text_processor: TranscriptTextProcessor,
        max_sessions: int,
        max_session_seconds: float,
        max_utterance_seconds: float,
        max_chunk_bytes: int,
        end_silence_ms: int,
        analysis_window_seconds: float,
        pre_roll_ms: int,
        partial_min_audio_seconds: float,
        partial_interval_seconds: float,
        privacy_ttl_seconds: int,
    ) -> None:
        self._detector = detector
        self._snapshot_store = snapshot_store
        self._recognizer = recognizer
        self._text_processor = text_processor
        self._max_sessions = max_sessions
        self._max_session_seconds = max_session_seconds
        self._max_utterance_seconds = max_utterance_seconds
        self._max_chunk_bytes = max_chunk_bytes
        self._end_silence_ms = end_silence_ms
        self._analysis_window_seconds = analysis_window_seconds
        self._pre_roll_ms = pre_roll_ms
        self._partial_min_audio_seconds = partial_min_audio_seconds
        self._partial_interval_seconds = partial_interval_seconds
        self._privacy_ttl_seconds = privacy_ttl_seconds
        self._active_sessions = 0
        self._capacity_lock = asyncio.Lock()

    @asynccontextmanager
    async def session(
        self,
        options: LiveTranscriptionOptions,
    ) -> AsyncIterator[LiveTranscriptionSession]:
        """Reserve capacity and guarantee in-memory audio cleanup on every exit."""

        async with self._capacity_lock:
            if self._active_sessions >= self._max_sessions:
                raise LiveCapacityExceededError(self._max_sessions)
            self._active_sessions += 1

        session = LiveTranscriptionSession(
            options=options,
            detector=self._detector,
            snapshot_store=self._snapshot_store,
            recognizer=self._recognizer,
            text_processor=self._text_processor,
            max_session_seconds=self._max_session_seconds,
            max_utterance_seconds=self._max_utterance_seconds,
            max_chunk_bytes=self._max_chunk_bytes,
            end_silence_ms=self._end_silence_ms,
            analysis_window_seconds=self._analysis_window_seconds,
            pre_roll_ms=self._pre_roll_ms,
            partial_min_audio_seconds=self._partial_min_audio_seconds,
            partial_interval_seconds=self._partial_interval_seconds,
            privacy_ttl_seconds=self._privacy_ttl_seconds,
        )
        try:
            yield session
        finally:
            await session.discard()
            async with self._capacity_lock:
                self._active_sessions -= 1


class LiveTranscriptionSession:
    """Detect utterances and emit best-effort partial plus accurate final text."""

    def __init__(
        self,
        *,
        options: LiveTranscriptionOptions,
        detector: VoiceActivityDetector,
        snapshot_store: PcmAudioSnapshotStore,
        recognizer: SpeechRecognizer,
        text_processor: TranscriptTextProcessor,
        max_session_seconds: float,
        max_utterance_seconds: float,
        max_chunk_bytes: int,
        end_silence_ms: int,
        analysis_window_seconds: float,
        pre_roll_ms: int,
        partial_min_audio_seconds: float,
        partial_interval_seconds: float,
        privacy_ttl_seconds: int,
    ) -> None:
        self._options = options
        self._detector = detector
        self._snapshot_store = snapshot_store
        self._recognizer = recognizer
        self._text_processor = text_processor
        self._max_session_samples = int(max_session_seconds * options.sample_rate_hz)
        self._max_session_seconds = max_session_seconds
        self._max_utterance_samples = int(max_utterance_seconds * options.sample_rate_hz)
        self._max_chunk_bytes = max_chunk_bytes
        self._end_silence_samples = int(end_silence_ms * options.sample_rate_hz / 1000)
        self._analysis_bytes = (
            int(analysis_window_seconds * options.sample_rate_hz) * _BYTES_PER_SAMPLE
        )
        self._pre_roll_bytes = int(pre_roll_ms * options.sample_rate_hz / 1000) * 2
        self._partial_min_samples = int(partial_min_audio_seconds * options.sample_rate_hz)
        self._partial_interval_samples = int(partial_interval_seconds * options.sample_rate_hz)
        self._privacy_ttl_seconds = privacy_ttl_seconds

        self._total_samples = 0
        self._last_vad_total_samples = 0
        self._pre_roll = bytearray()
        self._utterance = bytearray()
        self._utterance_start_sample = 0
        self._last_speech_sample: int | None = None
        self._last_partial_samples = 0
        self._last_partial_text = ""
        self._utterance_id = 0
        self._revision = 0
        self._speech_active = False
        self._closed = False

    async def ingest(self, pcm_chunk: bytes) -> tuple[LiveSessionEvent, ...]:
        """Consume one bounded PCM frame and emit zero or more ordered events."""

        self._require_open()
        if not pcm_chunk:
            return ()
        if len(pcm_chunk) > self._max_chunk_bytes or len(pcm_chunk) % _BYTES_PER_SAMPLE:
            raise ValueError("live PCM chunk is invalid or exceeds the configured limit")

        chunk_samples = len(pcm_chunk) // _BYTES_PER_SAMPLE
        if self._total_samples + chunk_samples > self._max_session_samples:
            raise LiveSessionLimitError(self._max_session_seconds)
        self._total_samples += chunk_samples

        if self._speech_active:
            self._utterance.extend(pcm_chunk)
        else:
            self._pre_roll.extend(pcm_chunk)
            self._trim_left(self._pre_roll, self._pre_roll_bytes)

        events: list[LiveSessionEvent] = []
        if self._total_samples - self._last_vad_total_samples >= _VAD_WINDOW_SAMPLES:
            self._last_vad_total_samples = self._total_samples
            analysis = self._analysis_audio()
            regions = await self._detector.detect(
                analysis,
                sample_rate_hz=self._options.sample_rate_hz,
            )
            analysis_start_sample = self._total_samples - len(analysis) // _BYTES_PER_SAMPLE
            if regions:
                latest_speech_sample = analysis_start_sample + int(
                    regions[-1].end_seconds * self._options.sample_rate_hz
                )
                self._last_speech_sample = max(
                    self._last_speech_sample or 0,
                    latest_speech_sample,
                )
                if not self._speech_active:
                    self._start_utterance()
                    events.append(self._speech_started_event())

        if not self._speech_active:
            return tuple(events)

        utterance_samples = len(self._utterance) // _BYTES_PER_SAMPLE
        if utterance_samples >= self._max_utterance_samples:
            final_event = await self._finalize(retain_silence_tail=False)
            if final_event is not None:
                events.append(final_event)
            return tuple(events)

        if (
            self._last_speech_sample is not None
            and self._total_samples - self._last_speech_sample >= self._end_silence_samples
        ):
            final_event = await self._finalize(retain_silence_tail=True)
            if final_event is not None:
                events.append(final_event)
            return tuple(events)

        if (
            utterance_samples >= self._partial_min_samples
            and utterance_samples - self._last_partial_samples >= self._partial_interval_samples
        ):
            partial_event = await self._partial()
            if partial_event is not None:
                events.append(partial_event)
        return tuple(events)

    async def commit(self) -> tuple[LiveSessionEvent, ...]:
        """Finalize the active utterance when the client signals its boundary."""

        self._require_open()
        if not self._speech_active:
            return ()
        event = await self._finalize(retain_silence_tail=False)
        return () if event is None else (event,)

    async def close(self) -> tuple[LiveSessionEvent, ...]:
        """Finalize active speech and permanently close the session."""

        if self._closed:
            return ()
        events = await self.commit() if self._speech_active else ()
        self._closed = True
        self._clear_audio()
        return events

    async def discard(self) -> None:
        """Drop all in-memory PCM without triggering inference."""

        self._closed = True
        self._clear_audio()

    def _start_utterance(self) -> None:
        self._speech_active = True
        self._utterance_id += 1
        self._revision = 0
        self._utterance = bytearray(self._pre_roll)
        self._utterance_start_sample = self._total_samples - len(self._pre_roll) // 2
        self._last_partial_samples = 0
        self._last_partial_text = ""

    def _speech_started_event(self) -> LiveSessionEvent:
        started_at = self._utterance_start_sample / self._options.sample_rate_hz
        return LiveSessionEvent(
            event_type=LiveEventType.SPEECH_STARTED,
            utterance_id=self._utterance_id,
            revision=0,
            start_seconds=started_at,
            end_seconds=started_at,
        )

    async def _partial(self) -> LiveSessionEvent | None:
        snapshot = bytes(self._utterance)
        snapshot_samples = len(snapshot) // _BYTES_PER_SAMPLE
        self._last_partial_samples = snapshot_samples
        try:
            result = await self._transcribe(snapshot, draft=True)
        except (InvalidAudioError, UncertainLanguageError):
            return None
        if result.text == self._last_partial_text:
            return None
        self._last_partial_text = result.text
        self._revision += 1
        return self._transcript_event(LiveEventType.PARTIAL, result, snapshot_samples)

    async def _finalize(self, *, retain_silence_tail: bool) -> LiveSessionEvent | None:
        snapshot = bytes(self._utterance)
        snapshot_samples = len(snapshot) // _BYTES_PER_SAMPLE
        retained_tail = snapshot[-self._pre_roll_bytes :] if retain_silence_tail else b""
        try:
            result = await self._transcribe(snapshot)
            self._revision += 1
            return self._transcript_event(LiveEventType.FINAL, result, snapshot_samples)
        finally:
            self._speech_active = False
            self._utterance = bytearray()
            self._pre_roll = bytearray(retained_tail)
            self._last_speech_sample = None
            self._last_partial_samples = 0
            self._last_partial_text = ""

    async def _transcribe(
        self,
        pcm_audio: bytes,
        *,
        draft: bool = False,
    ) -> TranscriptionResult:
        expires_at = datetime.now(tz=UTC) + timedelta(seconds=self._privacy_ttl_seconds)
        reference = await self._snapshot_store.create(
            pcm_audio,
            sample_rate_hz=self._options.sample_rate_hz,
            expires_at=expires_at,
        )
        try:
            result = await self._recognizer.transcribe(
                TranscriptionRequest(
                    audio=reference,
                    language=self._options.language,
                    vocabulary=self._options.vocabulary,
                    word_timestamps=self._options.word_timestamps,
                    draft=draft,
                )
            )
            processed = self._text_processor.process(result)
            return self._offset_result(
                processed,
                self._utterance_start_sample / self._options.sample_rate_hz,
            )
        finally:
            await self._snapshot_store.delete(reference)

    def _transcript_event(
        self,
        event_type: LiveEventType,
        result: TranscriptionResult,
        snapshot_samples: int,
    ) -> LiveSessionEvent:
        start_seconds = self._utterance_start_sample / self._options.sample_rate_hz
        return LiveSessionEvent(
            event_type=event_type,
            utterance_id=self._utterance_id,
            revision=self._revision,
            start_seconds=start_seconds,
            end_seconds=start_seconds + snapshot_samples / self._options.sample_rate_hz,
            result=result,
        )

    def _analysis_audio(self) -> bytes:
        source = self._utterance if self._speech_active else self._pre_roll
        return bytes(source[-self._analysis_bytes :])

    def _clear_audio(self) -> None:
        self._pre_roll.clear()
        self._utterance.clear()
        self._speech_active = False
        self._last_speech_sample = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("live transcription session is closed")

    @staticmethod
    def _trim_left(buffer: bytearray, max_bytes: int) -> None:
        if len(buffer) > max_bytes:
            del buffer[: len(buffer) - max_bytes]

    @staticmethod
    def _offset_result(result: TranscriptionResult, offset: float) -> TranscriptionResult:
        if offset == 0:
            return result
        return replace(
            result,
            segments=tuple(
                replace(
                    segment,
                    start_seconds=segment.start_seconds + offset,
                    end_seconds=segment.end_seconds + offset,
                    words=tuple(
                        replace(
                            word,
                            start_seconds=word.start_seconds + offset,
                            end_seconds=word.end_seconds + offset,
                        )
                        for word in segment.words
                    ),
                )
                for segment in result.segments
            ),
        )
