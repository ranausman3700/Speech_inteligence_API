"""Native-script transcript post-processing tests."""

from __future__ import annotations

from speech_intelligence_api.adapters.native_text import NativeTranscriptTextProcessor
from speech_intelligence_api.domain.enums import ChineseScript, LanguageCode
from speech_intelligence_api.domain.models import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)


def _result(
    text: str,
    *,
    language: LanguageCode,
    chinese_script: ChineseScript | None = None,
) -> TranscriptionResult:
    word = TranscriptWord(
        text=text,
        start_seconds=0,
        end_seconds=1,
        confidence_estimate=0.9,
    )
    segment = TranscriptSegment(
        text=text,
        start_seconds=0,
        end_seconds=1,
        language=language,
        words=(word,),
        confidence_estimate=0.8,
    )
    return TranscriptionResult(
        language=language,
        language_confidence_estimate=0.95,
        text=text,
        segments=(segment,),
        chinese_script=chinese_script,
    )


def test_converts_chinese_to_requested_traditional_script() -> None:
    processor = NativeTranscriptTextProcessor()

    result = processor.process(
        _result(
            "汉语",
            language=LanguageCode.CHINESE,
            chinese_script=ChineseScript.TRADITIONAL,
        )
    )

    assert result.text == "漢語"
    assert result.segments[0].text == "漢語"
    assert result.segments[0].words[0].text == "漢語"
    assert result.chinese_script is ChineseScript.TRADITIONAL


def test_converts_chinese_to_requested_simplified_script() -> None:
    processor = NativeTranscriptTextProcessor()

    result = processor.process(
        _result(
            "漢語",
            language=LanguageCode.CHINESE,
            chinese_script=ChineseScript.SIMPLIFIED,
        )
    )

    assert result.text == "汉语"


def test_preserves_arabic_script_while_normalizing_unsafe_text() -> None:
    processor = NativeTranscriptTextProcessor()

    result = processor.process(
        _result(
            "  السَّلَامُ\t عَلَيْكُمْ\r\n\x00  ",
            language=LanguageCode.ARABIC,
        )
    )

    assert result.text == "السَّلَامُ عَلَيْكُمْ"
    assert "\x00" not in result.segments[0].text
    assert result.language is LanguageCode.ARABIC
