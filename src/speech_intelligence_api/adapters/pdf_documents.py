"""PDF export renderer with per-script font selection.

A PDF must embed a font that actually contains the glyphs it draws, and Arabic,
Devanagari, Bengali and Thai additionally need shaping so letters join and stack
correctly. This renderer therefore picks a Noto font from a configured directory
by inspecting the scripts present in the text, and enables HarfBuzz shaping.

Fonts are deliberately not vendored: an operator installs only the scripts their
callers actually use, which keeps the image small. When no bundled font covers
the text the renderer raises, and the caller is told to export plain text
instead, which always works.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any

from speech_intelligence_api.config import Settings
from speech_intelligence_api.domain.errors import ErrorCode, ServiceError
from speech_intelligence_api.ports.documents import RenderedDocument

logger = logging.getLogger(__name__)

_PAGE_MARGIN_MM = 15
_TITLE_SIZE_PT = 16
_BODY_SIZE_PT = 11
_LINE_HEIGHT_MM = 6

#: Unicode script keywords mapped to the Noto font file that covers them. Latin
#: is last so it only wins when no complex script appears in the text.
_FONT_BY_SCRIPT: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ARABIC",), "NotoSansArabic-Regular.ttf"),
    (("HEBREW",), "NotoSansHebrew-Regular.ttf"),
    (("DEVANAGARI",), "NotoSansDevanagari-Regular.ttf"),
    (("BENGALI",), "NotoSansBengali-Regular.ttf"),
    (("THAI",), "NotoSansThai-Regular.ttf"),
    (("CJK", "HIRAGANA", "KATAKANA", "HANGUL"), "NotoSansCJK-Regular.ttf"),
    (("LATIN", "CYRILLIC", "GREEK"), "NotoSans-Regular.ttf"),
)

#: Right-to-left scripts need the paragraph aligned to the right edge.
_RTL_SCRIPTS = ("ARABIC", "HEBREW")


class PdfDocumentRenderer:
    """Render a titled transcript as a single-column PDF."""

    def __init__(self, settings: Settings) -> None:
        self._font_root = settings.export_font_root

    def render(self, *, title: str, body_lines: tuple[str, ...], filename: str) -> RenderedDocument:
        text = "\n".join((title, *body_lines))
        font_path = self._font_for(text)

        pdf = self._new_document(font_path)
        if title:
            pdf.set_font("export", size=_TITLE_SIZE_PT)
            pdf.multi_cell(0, _LINE_HEIGHT_MM, title, align=self._align(title))
            pdf.ln(_LINE_HEIGHT_MM / 2)

        pdf.set_font("export", size=_BODY_SIZE_PT)
        for line in body_lines:
            pdf.multi_cell(0, _LINE_HEIGHT_MM, line, align=self._align(line))
            pdf.ln(_LINE_HEIGHT_MM / 3)

        return RenderedDocument(
            content=bytes(pdf.output()),
            media_type="application/pdf",
            filename=f"{filename}.pdf",
        )

    def _new_document(self, font_path: Path) -> Any:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_margins(_PAGE_MARGIN_MM, _PAGE_MARGIN_MM, _PAGE_MARGIN_MM)
        pdf.set_auto_page_break(auto=True, margin=_PAGE_MARGIN_MM)
        pdf.add_page()
        pdf.add_font("export", "", str(font_path))
        # Joins Arabic letters and stacks Devanagari/Thai marks correctly.
        pdf.set_text_shaping(True)
        return pdf

    def _font_for(self, text: str) -> Path:
        if self._font_root is None:
            raise self._unavailable("no export font directory is configured")

        scripts = _scripts_in(text)
        for keywords, filename in _FONT_BY_SCRIPT:
            if not scripts.intersection(keywords):
                continue
            candidate = self._font_root / filename
            if candidate.is_file():
                return candidate
            logger.warning(
                "PDF export font missing for a requested script",
                extra={"export_font": filename},
            )
            raise self._unavailable(f"no installed font covers this text ({filename} missing)")

        fallback = self._font_root / _FONT_BY_SCRIPT[-1][1]
        if fallback.is_file():
            return fallback
        raise self._unavailable("no installed font covers this text")

    @staticmethod
    def _align(line: str) -> str:
        return "R" if _scripts_in(line).intersection(_RTL_SCRIPTS) else "L"

    @staticmethod
    def _unavailable(detail: str) -> ServiceError:
        return ServiceError(
            ErrorCode.INVALID_REQUEST,
            f"PDF export is unavailable for this text: {detail}. Export txt instead.",
        )


def _scripts_in(text: str) -> frozenset[str]:
    """Return the Unicode script keywords present in the text."""

    found: set[str] = set()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for keywords, _ in _FONT_BY_SCRIPT:
            for keyword in keywords:
                if keyword in name:
                    found.add(keyword)
    return frozenset(found)
