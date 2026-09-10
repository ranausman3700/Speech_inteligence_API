"""Pure domain policy tests."""

from datetime import UTC, datetime, timedelta

import pytest

from speech_intelligence_api.domain.enums import (
    SUPPORTED_LANGUAGE_VARIANTS,
    ChineseScript,
    JobKind,
    JobQueue,
    JobStatus,
    LanguageCode,
    LanguageSelectionMode,
    LiveEventType,
)
from speech_intelligence_api.domain.jobs import (
    ConversationJobPayload,
    ConversationJobResult,
    TranscriptionJobPayload,
    TranscriptionJobResult,
)
from speech_intelligence_api.domain.models import (
    BlobReference,
    ConversationResult,
    DiarizationTurn,
    LanguageSelection,
    LiveSessionEvent,
    LiveTranscriptionOptions,
    SpeechRegion,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from tests.factories import make_job


def test_supported_language_contract_covers_every_advertised_language() -> None:
    # Pinned because automatic detection filters candidates to exactly this set:
    # a code missing here is silently undetectable rather than merely unlisted.
    assert {code.value for code in LanguageCode} == {
        "ar",
        "bn",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sv",
        "th",
        "tr",
        "uk",
        "ur",
        "vi",
        "zh",
    }
    assert len(SUPPORTED_LANGUAGE_VARIANTS) == 35
    assert len({variant.code for variant in SUPPORTED_LANGUAGE_VARIANTS}) == 34
    assert {variant.code for variant in SUPPORTED_LANGUAGE_VARIANTS} == set(LanguageCode)
    chinese_scripts = {
        variant.chinese_script
        for variant in SUPPORTED_LANGUAGE_VARIANTS
        if variant.code is LanguageCode.CHINESE
    }
    assert chinese_scripts == {ChineseScript.SIMPLIFIED, ChineseScript.TRADITIONAL}


def test_explicit_mode_requires_language_and_chinese_script() -> None:
    with pytest.raises(ValueError, match="requires a language"):
        LanguageSelection(mode=LanguageSelectionMode.EXPLICIT)

    with pytest.raises(ValueError, match="requires a script"):
        LanguageSelection(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.CHINESE,
        )


def test_automatic_mode_rejects_forced_language() -> None:
    with pytest.raises(ValueError, match="cannot force"):
        LanguageSelection(
            mode=LanguageSelectionMode.AUTOMATIC,
            language=LanguageCode.ENGLISH,
        )


def test_native_unicode_text_remains_unchanged() -> None:
    word = TranscriptWord("السلام", 0.0, 0.5, 0.95)
    segment = TranscriptSegment(
        text="السلام عليكم",
        start_seconds=0.0,
        end_seconds=1.0,
        language=LanguageCode.ARABIC,
        words=(word,),
    )

    assert segment.text == "السلام عليكم"
    assert segment.words[0].text == "السلام"


def test_word_must_fall_inside_segment() -> None:
    word = TranscriptWord("outside", 0.0, 2.0)

    with pytest.raises(ValueError, match="within their segment"):
        TranscriptSegment(
            text="outside",
            start_seconds=0.5,
            end_seconds=1.5,
            language=LanguageCode.ENGLISH,
            words=(word,),
        )


def test_blob_reference_rejects_traversal() -> None:
    created_at = datetime.now(tz=UTC)

    with pytest.raises(ValueError, match="opaque path"):
        BlobReference(
            key="../secret",
            media_type="audio/wav",
            size_bytes=1,
            created_at=created_at,
            expires_at=created_at + timedelta(minutes=1),
        )


def test_job_state_machine_accepts_only_legal_transitions() -> None:
    running = make_job().transition(JobStatus.RUNNING, progress_percent=10)
    succeeded = running.transition(JobStatus.SUCCEEDED, progress_percent=100)

    assert succeeded.status is JobStatus.SUCCEEDED
    assert succeeded.progress_percent == 100
    with pytest.raises(ValueError, match="illegal job transition"):
        succeeded.transition(JobStatus.RUNNING)


def test_job_terminal_state_policy() -> None:
    assert JobStatus.SUCCEEDED.terminal is True
    assert JobStatus.RUNNING.terminal is False
    assert JobStatus.QUEUED.can_transition_to(JobStatus.CANCELLED) is True
    assert JobStatus.EXPIRED.can_transition_to(JobStatus.RUNNING) is False


def test_job_lifetime_cannot_exceed_privacy_limit() -> None:
    job = make_job()

    with pytest.raises(ValueError, match="cannot exceed 30 minutes"):
        type(job)(
            job_id=job.job_id,
            kind=job.kind,
            status=job.status,
            created_at=job.created_at,
            expires_at=job.created_at + timedelta(minutes=31),
        )


def test_blob_reference_enforces_size_time_and_expiry() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    def build_blob(
        *,
        size_bytes: int = 1,
        created: datetime = created_at,
        expires: datetime = created_at + timedelta(minutes=1),
    ) -> BlobReference:
        return BlobReference(
            key="audio/blob.wav",
            media_type="audio/wav",
            size_bytes=size_bytes,
            created_at=created,
            expires_at=expires,
        )

    with pytest.raises(ValueError, match="size cannot be negative"):
        build_blob(size_bytes=-1)
    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        build_blob(created=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="expiry must be after"):
        build_blob(expires=created_at)
    with pytest.raises(ValueError, match="lifetime cannot exceed 30 minutes"):
        build_blob(expires=created_at + timedelta(minutes=31))


def test_language_selection_rejects_non_chinese_script_option() -> None:
    with pytest.raises(ValueError, match="only be set for Chinese"):
        LanguageSelection(
            mode=LanguageSelectionMode.EXPLICIT,
            language=LanguageCode.ENGLISH,
            chinese_script=ChineseScript.TRADITIONAL,
        )


def test_transcription_request_bounds_custom_vocabulary() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    blob = BlobReference(
        key="audio/input.wav",
        media_type="audio/wav",
        size_bytes=10,
        created_at=created_at,
        expires_at=created_at + timedelta(minutes=1),
    )
    selection = LanguageSelection(
        mode=LanguageSelectionMode.EXPLICIT,
        language=LanguageCode.ENGLISH,
    )

    with pytest.raises(ValueError, match="cannot exceed 100"):
        TranscriptionRequest(blob, selection, vocabulary=("term",) * 101)
    with pytest.raises(ValueError, match="1 to 100 characters"):
        TranscriptionRequest(blob, selection, vocabulary=(" ",))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text": ""}, "text cannot be empty"),
        ({"start_seconds": -1.0}, "non-negative and monotonic"),
        ({"confidence_estimate": 1.1}, "between 0 and 1"),
    ],
)
def test_transcript_word_validation(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "text": "word",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "confidence_estimate": 0.8,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TranscriptWord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text": ""}, "text cannot be empty"),
        ({"end_seconds": -1.0}, "non-negative and monotonic"),
        ({"confidence_estimate": -0.1}, "between 0 and 1"),
    ],
)
def test_transcript_segment_validation(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "text": "segment",
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "language": LanguageCode.ENGLISH,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TranscriptSegment(**values)  # type: ignore[arg-type]


def test_transcription_result_enforces_language_and_script() -> None:
    with pytest.raises(ValueError, match="language confidence"):
        TranscriptionResult(LanguageCode.ENGLISH, 1.1, "", ())
    with pytest.raises(ValueError, match="Chinese results"):
        TranscriptionResult(LanguageCode.CHINESE, 0.9, "中文", ())
    with pytest.raises(ValueError, match="non-Chinese"):
        TranscriptionResult(
            LanguageCode.ENGLISH,
            0.9,
            "text",
            (),
            chinese_script=ChineseScript.SIMPLIFIED,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SpeechRegion(1.0, 1.0, 0.9), "positive and monotonic"),
        (lambda: SpeechRegion(0.0, 1.0, 1.1), "between 0 and 1"),
        (lambda: DiarizationTurn("", 0.0, 1.0), "label cannot be empty"),
        (lambda: DiarizationTurn("Person 1", 1.0, 1.0), "positive and monotonic"),
        (lambda: DiarizationTurn("Person 1", 0.0, 1.0, -0.1), "between 0 and 1"),
    ],
)
def test_timed_domain_value_validation(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_live_options_enforce_pcm_and_vocabulary_contract() -> None:
    language = LanguageSelection(
        mode=LanguageSelectionMode.EXPLICIT,
        language=LanguageCode.ENGLISH,
    )

    with pytest.raises(ValueError, match="16 kHz"):
        LiveTranscriptionOptions(language=language, sample_rate_hz=8_000)
    with pytest.raises(ValueError, match="cannot exceed"):
        LiveTranscriptionOptions(language=language, vocabulary=tuple("term" for _ in range(101)))
    with pytest.raises(ValueError, match="1 to 100"):
        LiveTranscriptionOptions(language=language, vocabulary=("",))


def test_live_events_enforce_type_result_and_revision_contract() -> None:
    with pytest.raises(ValueError, match="positive"):
        LiveSessionEvent(LiveEventType.SPEECH_STARTED, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="revision"):
        LiveSessionEvent(LiveEventType.SPEECH_STARTED, 1, -1, 0, 0)
    with pytest.raises(ValueError, match="timestamps"):
        LiveSessionEvent(LiveEventType.SPEECH_STARTED, 1, 0, 1, 0)
    with pytest.raises(ValueError, match="transcription result"):
        LiveSessionEvent(LiveEventType.FINAL, 1, 1, 0, 1)
    with pytest.raises(ValueError, match="revision zero"):
        LiveSessionEvent(LiveEventType.SPEECH_STARTED, 1, 1, 0, 0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"job_id": ""}, "ID cannot be empty"),
        ({"created_at": datetime(2026, 1, 1)}, "created_at must be timezone-aware"),
        ({"expires_at": datetime(2026, 1, 1, tzinfo=UTC)}, "expiry must be after"),
        ({"progress_percent": 101}, "progress must be between"),
    ],
)
def test_job_record_validation(changes: dict[str, object], message: str) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    values: dict[str, object] = {
        "job_id": "job_1",
        "kind": JobKind.TRANSCRIPTION,
        "status": JobStatus.QUEUED,
        "created_at": created_at,
        "expires_at": created_at + timedelta(minutes=30),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        type(make_job())(**values)  # type: ignore[arg-type]


def test_conversation_domain_enforces_speaker_and_script_contracts() -> None:
    segment = TranscriptSegment("hello", 0, 1, LanguageCode.ENGLISH, speaker="Person 1")
    valid = ConversationResult(
        LanguageCode.ENGLISH,
        0.9,
        None,
        "hello",
        "Person 1: hello",
        (segment,),
    )
    assert valid.segments == (segment,)

    with pytest.raises(ValueError, match="speaker metadata"):
        TranscriptSegment(
            "hello",
            0,
            1,
            LanguageCode.ENGLISH,
            speaker_uncertain=True,
        )
    with pytest.raises(ValueError, match="speaker confidence"):
        TranscriptSegment(
            "hello",
            0,
            1,
            LanguageCode.ENGLISH,
            speaker="Person 1",
            speaker_confidence_estimate=1.1,
        )
    with pytest.raises(ValueError, match="language confidence"):
        ConversationResult(LanguageCode.ENGLISH, 1.1, None, "hello", "text", (segment,))
    with pytest.raises(ValueError, match="identify their output script"):
        ConversationResult(LanguageCode.CHINESE, 0.9, None, "中文", "Person 1: 中文", (segment,))
    with pytest.raises(ValueError, match="non-Chinese"):
        ConversationResult(
            LanguageCode.ENGLISH,
            0.9,
            ChineseScript.SIMPLIFIED,
            "hello",
            "Person 1: hello",
            (segment,),
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        ConversationResult(LanguageCode.ENGLISH, 0.9, None, "", "", (segment,))
    with pytest.raises(ValueError, match="speaker labels"):
        ConversationResult(
            LanguageCode.ENGLISH,
            0.9,
            None,
            "hello",
            "Unknown: hello",
            (TranscriptSegment("hello", 0, 1, LanguageCode.ENGLISH),),
        )


def test_job_payloads_and_results_enforce_queue_and_duration_contracts() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    blob = BlobReference(
        "audio.wav",
        "audio/wav",
        1,
        created_at,
        created_at + timedelta(minutes=5),
    )
    request = TranscriptionRequest(
        blob,
        LanguageSelection(LanguageSelectionMode.EXPLICIT, LanguageCode.ENGLISH),
    )
    transcript = TranscriptionResult(LanguageCode.ENGLISH, 1, "hello", ())
    conversation = ConversationResult(
        LanguageCode.ENGLISH,
        1,
        None,
        "hello",
        "Person 1: hello",
        (TranscriptSegment("hello", 0, 1, LanguageCode.ENGLISH, speaker="Person 1"),),
    )

    with pytest.raises(ValueError, match="audio duration"):
        TranscriptionJobPayload(request, 0, JobQueue.SHORT_TRANSCRIPTION)
    with pytest.raises(ValueError, match="transcription queue"):
        TranscriptionJobPayload(request, 1, JobQueue.DIARIZATION)
    with pytest.raises(ValueError, match="result audio duration"):
        TranscriptionJobResult(transcript, 0)
    with pytest.raises(ValueError, match="conversation audio duration"):
        ConversationJobPayload(request, 0, 2, 0.6)
    with pytest.raises(ValueError, match="expected speakers"):
        ConversationJobPayload(request, 1, 1, 0.6)
    with pytest.raises(ValueError, match="confidence threshold"):
        ConversationJobPayload(request, 1, 2, 1.1)
    with pytest.raises(ValueError, match="diarization queue"):
        ConversationJobPayload(request, 1, 2, 0.6, JobQueue.LONG_AUDIO)
    with pytest.raises(ValueError, match="conversation result audio duration"):
        ConversationJobResult(conversation, 0)
