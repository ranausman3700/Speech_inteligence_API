"""Unicode-safe transcript formatting and deterministic Chinese conversion."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from opencc import OpenCC

from speech_intelligence_api.domain.enums import ChineseScript, LanguageCode
from speech_intelligence_api.domain.models import TranscriptionResult

_HORIZONTAL_SPACE = re.compile(r"[^\S\r\n]+")
_NEWLINE_SPACE = re.compile(r" *\n *")


class NativeTranscriptTextProcessor:
    """Preserve source script while removing unsafe control characters."""

    def __init__(self) -> None:
        self._to_simplified = OpenCC("t2s")
        self._to_traditional = OpenCC("s2t")

    def process(self, result: TranscriptionResult) -> TranscriptionResult:
        """Normalize Unicode and convert Chinese only to the requested script."""

        converter = None
        if result.language is LanguageCode.CHINESE:
            converter = (
                self._to_traditional
                if result.chinese_script is ChineseScript.TRADITIONAL
                else self._to_simplified
            )

        def process_text(value: str) -> str:
            safe_value = self._normalize_safe_unicode(value)
            return converter.convert(safe_value) if converter is not None else safe_value

        segments = tuple(
            replace(
                segment,
                text=process_text(segment.text),
                words=tuple(replace(word, text=process_text(word.text)) for word in segment.words),
            )
            for segment in result.segments
        )
        return replace(
            result,
            text=process_text(result.text),
            segments=segments,
        )

    @staticmethod
    def _normalize_safe_unicode(value: str) -> str:
        normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
        without_controls = "".join(
            character
            for character in normalized
            if character in {"\n", "\t"} or ord(character) >= 32
        )
        compact = _HORIZONTAL_SPACE.sub(" ", without_controls)
        return _NEWLINE_SPACE.sub("\n", compact).strip()
