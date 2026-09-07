"""Deterministic word-to-speaker alignment and conversation formatting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from speech_intelligence_api.domain.enums import LanguageCode
from speech_intelligence_api.domain.models import (
    ConversationResult,
    DiarizationTurn,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)

_UNKNOWN_SPEAKER = "Unknown"
_NO_INTERWORD_SPACE_LANGUAGES = frozenset(
    {
        LanguageCode.CHINESE,
        LanguageCode.JAPANESE,
        LanguageCode.THAI,
    }
)
_NO_SPACE_BEFORE = frozenset(
    ",.!?;:%)]}\u00bb\u2019\u201d\u2026\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u3011\u300b\u300d\u300f\uff05"
)
_NO_SPACE_AFTER = frozenset("([{\u00ab\u2018\u201c\uff08\u3010\u300a\u300c\u300e")
_ATTACHING_PREFIXES = ("'", "\u2019", "-", "\u2010", "\u2011", "/")
_ATTACHING_SUFFIXES = ("-", "\u2010", "\u2011", "/")


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    start_seconds: float
    end_seconds: float
    language: LanguageCode
    word: TranscriptWord | None
    transcript_confidence: float | None


@dataclass(frozen=True, slots=True)
class _AssignedToken:
    token: _Token
    speaker: str
    speaker_confidence: float
    uncertain: bool
    overlapping: bool


class ConversationAssembler:
    """Combine timestamped ASR with exclusive speaker turns."""

    def assemble(
        self,
        transcript: TranscriptionResult,
        turns: tuple[DiarizationTurn, ...],
        *,
        speaker_confidence_threshold: float,
    ) -> ConversationResult:
        if not transcript.text.strip() or not transcript.segments:
            raise ValueError("conversation transcription cannot be empty")
        if not turns:
            raise ValueError("speaker diarization returned no speech turns")
        assigned = tuple(
            self._assign(token, turns, speaker_confidence_threshold)
            for token in self._tokens(transcript)
        )
        labels = self._speaker_labels(assigned)
        segments = self._segments(assigned, labels)
        formatted = "\n".join(self._formatted_line(segment) for segment in segments)
        return ConversationResult(
            language=transcript.language,
            language_confidence_estimate=transcript.language_confidence_estimate,
            chinese_script=transcript.chinese_script,
            raw_transcript=transcript.text.strip(),
            formatted_transcript=formatted,
            segments=segments,
        )

    @staticmethod
    def _tokens(transcript: TranscriptionResult) -> tuple[_Token, ...]:
        tokens: list[_Token] = []
        for segment in transcript.segments:
            if segment.words:
                tokens.extend(
                    _Token(
                        text=word.text,
                        start_seconds=word.start_seconds,
                        end_seconds=word.end_seconds,
                        language=segment.language,
                        word=word,
                        transcript_confidence=word.confidence_estimate,
                    )
                    for word in segment.words
                )
            else:
                tokens.append(
                    _Token(
                        text=segment.text,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        language=segment.language,
                        word=None,
                        transcript_confidence=segment.confidence_estimate,
                    )
                )
        return tuple(tokens)

    @staticmethod
    def _assign(
        token: _Token,
        turns: tuple[DiarizationTurn, ...],
        threshold: float,
    ) -> _AssignedToken:
        duration = token.end_seconds - token.start_seconds
        scored: list[tuple[float, DiarizationTurn]] = []
        for turn in turns:
            overlap = max(
                0.0,
                min(token.end_seconds, turn.end_seconds)
                - max(token.start_seconds, turn.start_seconds),
            )
            if overlap > 0:
                scored.append((overlap, turn))
        if not scored and duration == 0:
            scored = [
                (1.0, turn)
                for turn in turns
                if turn.start_seconds <= token.start_seconds <= turn.end_seconds
            ]
        if not scored:
            return _AssignedToken(token, _UNKNOWN_SPEAKER, 0, True, False)
        overlap, selected = max(scored, key=lambda item: item[0])
        confidence = 1.0 if duration == 0 else min(1.0, overlap / duration)
        uncertain = confidence < threshold or selected.overlapping_speech
        return _AssignedToken(
            token,
            selected.speaker,
            confidence,
            uncertain,
            selected.overlapping_speech,
        )

    @staticmethod
    def _speaker_labels(assigned: tuple[_AssignedToken, ...]) -> dict[str, str]:
        labels: dict[str, str] = {_UNKNOWN_SPEAKER: _UNKNOWN_SPEAKER}
        for item in assigned:
            if item.speaker not in labels:
                labels[item.speaker] = f"Person {len(labels)}"
        return labels

    @classmethod
    def _segments(
        cls,
        assigned: tuple[_AssignedToken, ...],
        labels: dict[str, str],
    ) -> tuple[TranscriptSegment, ...]:
        groups: list[list[_AssignedToken]] = []
        for item in assigned:
            if not groups or not cls._same_group(groups[-1][-1], item):
                groups.append([item])
            else:
                groups[-1].append(item)
        return tuple(cls._segment(group, labels[group[0].speaker]) for group in groups)

    @staticmethod
    def _same_group(previous: _AssignedToken, current: _AssignedToken) -> bool:
        return (
            previous.speaker == current.speaker
            and previous.token.language is current.token.language
            and current.token.start_seconds - previous.token.end_seconds <= 1.5
        )

    @staticmethod
    def _segment(group: list[_AssignedToken], speaker: str) -> TranscriptSegment:
        words = tuple(item.token.word for item in group if item.token.word is not None)
        transcript_scores = [
            item.token.transcript_confidence
            for item in group
            if item.token.transcript_confidence is not None
        ]
        return TranscriptSegment(
            text=ConversationAssembler._join_token_texts(
                (item.token.text for item in group),
                group[0].token.language,
            ),
            start_seconds=group[0].token.start_seconds,
            end_seconds=group[-1].token.end_seconds,
            language=group[0].token.language,
            words=words,
            speaker=speaker,
            speaker_uncertain=any(item.uncertain for item in group),
            speaker_confidence_estimate=fmean(item.speaker_confidence for item in group),
            overlapping_speech=any(item.overlapping for item in group),
            confidence_estimate=fmean(transcript_scores) if transcript_scores else None,
        )

    @staticmethod
    def _join_token_texts(texts: Iterable[str], language: LanguageCode) -> str:
        """Rebuild readable text from normalized ASR words without harming native scripts."""

        tokens = tuple(text.strip() for text in texts if text.strip())
        if not tokens:
            raise ValueError("conversation segment cannot contain only empty tokens")
        if language in _NO_INTERWORD_SPACE_LANGUAGES:
            return "".join(tokens)

        result = tokens[0]
        previous = tokens[0]
        for token in tokens[1:]:
            separator = " "
            if (
                token[0] in _NO_SPACE_BEFORE
                or previous[-1] in _NO_SPACE_AFTER
                or token.startswith(_ATTACHING_PREFIXES)
                or previous.endswith(_ATTACHING_SUFFIXES)
            ):
                separator = ""
            result += f"{separator}{token}"
            previous = token
        return result

    @staticmethod
    def _formatted_line(segment: TranscriptSegment) -> str:
        suffix = " [overlap]" if segment.overlapping_speech else ""
        return f"{segment.speaker}{suffix}: {segment.text}"
