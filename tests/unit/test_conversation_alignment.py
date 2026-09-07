"""Deterministic transcript and speaker-turn alignment tests."""

from __future__ import annotations

import pytest

from speech_intelligence_api.application.conversation_alignment import ConversationAssembler
from speech_intelligence_api.domain.enums import ChineseScript, LanguageCode
from speech_intelligence_api.domain.models import (
    DiarizationTurn,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)


def _transcript(*, words: bool = True) -> TranscriptionResult:
    segment_words = (
        (
            TranscriptWord("Hello ", 0.0, 0.5, 0.9),
            TranscriptWord("there", 0.5, 1.0, 0.8),
            TranscriptWord("Hi", 1.1, 1.5, 0.7),
        )
        if words
        else ()
    )
    return TranscriptionResult(
        language=LanguageCode.ENGLISH,
        language_confidence_estimate=0.95,
        text="Hello there Hi",
        segments=(
            TranscriptSegment(
                text="Hello there Hi",
                start_seconds=0,
                end_seconds=1.5,
                language=LanguageCode.ENGLISH,
                words=segment_words,
                confidence_estimate=0.85,
            ),
        ),
    )


def test_assigns_stable_person_labels_and_preserves_raw_text() -> None:
    result = ConversationAssembler().assemble(
        _transcript(),
        (
            DiarizationTurn("SPEAKER_09", 0, 1),
            DiarizationTurn("SPEAKER_02", 1, 1.6),
        ),
        speaker_confidence_threshold=0.6,
    )

    assert result.raw_transcript == "Hello there Hi"
    assert result.formatted_transcript == "Person 1: Hello there\nPerson 2: Hi"
    assert [segment.speaker for segment in result.segments] == ["Person 1", "Person 2"]
    assert result.segments[0].confidence_estimate == pytest.approx(0.85)
    assert result.segments[0].speaker_confidence_estimate == 1


def test_marks_partial_assignments_overlap_and_unknown_speech_as_uncertain() -> None:
    result = ConversationAssembler().assemble(
        _transcript(),
        (
            DiarizationTurn("speaker-a", 0, 0.25),
            DiarizationTurn("speaker-b", 1.1, 1.5, overlapping_speech=True),
        ),
        speaker_confidence_threshold=0.6,
    )

    assert [segment.speaker for segment in result.segments] == [
        "Person 1",
        "Unknown",
        "Person 2",
    ]
    assert all(segment.speaker_uncertain for segment in result.segments)
    assert result.segments[-1].overlapping_speech is True
    assert "Person 2 [overlap]: Hi" in result.formatted_transcript
    assert "uncertain" not in result.formatted_transcript


def test_merges_adjacent_speaker_fragments_across_uncertainty_boundaries() -> None:
    result = ConversationAssembler().assemble(
        _transcript(),
        (
            DiarizationTurn("speaker-a", 0, 0.25),
            DiarizationTurn("speaker-a", 0.5, 1),
            DiarizationTurn("speaker-b", 1, 1.6),
        ),
        speaker_confidence_threshold=0.6,
    )

    assert result.formatted_transcript == "Person 1: Hello there\nPerson 2: Hi"
    assert len(result.segments) == 2
    assert result.segments[0].speaker_uncertain is True


def test_restores_spaces_without_adding_space_before_punctuation() -> None:
    transcript = TranscriptionResult(
        language=LanguageCode.ENGLISH,
        language_confidence_estimate=1,
        text="Hello, world!",
        segments=(
            TranscriptSegment(
                text="Hello, world!",
                start_seconds=0,
                end_seconds=1,
                language=LanguageCode.ENGLISH,
                words=(
                    TranscriptWord("Hello", 0, 0.2),
                    TranscriptWord(",", 0.2, 0.3),
                    TranscriptWord("world", 0.3, 0.8),
                    TranscriptWord("!", 0.8, 1),
                ),
            ),
        ),
    )

    result = ConversationAssembler().assemble(
        transcript,
        (DiarizationTurn("speaker-a", 0, 1),),
        speaker_confidence_threshold=0.6,
    )

    assert result.formatted_transcript == "Person 1: Hello, world!"


def test_preserves_unspaced_native_scripts() -> None:
    transcript = TranscriptionResult(
        language=LanguageCode.CHINESE,
        language_confidence_estimate=1,
        text="\u4f60\u597d\u3002",
        segments=(
            TranscriptSegment(
                text="\u4f60\u597d\u3002",
                start_seconds=0,
                end_seconds=1,
                language=LanguageCode.CHINESE,
                words=(
                    TranscriptWord("\u4f60", 0, 0.3),
                    TranscriptWord("\u597d", 0.3, 0.8),
                    TranscriptWord("\u3002", 0.8, 1),
                ),
            ),
        ),
        chinese_script=ChineseScript.SIMPLIFIED,
    )

    result = ConversationAssembler().assemble(
        transcript,
        (DiarizationTurn("speaker-a", 0, 1),),
        speaker_confidence_threshold=0.6,
    )

    assert result.formatted_transcript == "Person 1: \u4f60\u597d\u3002"


def test_supports_segment_level_alignment_without_word_timestamps() -> None:
    result = ConversationAssembler().assemble(
        _transcript(words=False),
        (DiarizationTurn("speaker-a", 0, 1.5),),
        speaker_confidence_threshold=0.6,
    )

    assert result.segments[0].text == "Hello there Hi"
    assert result.segments[0].words == ()
    assert result.segments[0].confidence_estimate == 0.85


@pytest.mark.parametrize(
    ("transcript", "turns"),
    [
        (TranscriptionResult(LanguageCode.ENGLISH, 1, "", ()), (DiarizationTurn("a", 0, 1),)),
        (_transcript(), ()),
    ],
)
def test_rejects_empty_transcription_or_diarization(
    transcript: TranscriptionResult,
    turns: tuple[DiarizationTurn, ...],
) -> None:
    with pytest.raises(ValueError):
        ConversationAssembler().assemble(
            transcript,
            turns,
            speaker_confidence_threshold=0.6,
        )
