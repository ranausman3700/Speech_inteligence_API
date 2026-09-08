"""Plain-text export renderer.

UTF-8 text needs no font or shaping engine, so this renderer works identically
for every supported language and costs only the encode.
"""

from __future__ import annotations

from speech_intelligence_api.ports.documents import RenderedDocument


class PlainTextDocumentRenderer:
    """Write a title and body as UTF-8 text."""

    def render(self, *, title: str, body_lines: tuple[str, ...], filename: str) -> RenderedDocument:
        blocks = [title, "=" * len(title), "", *body_lines] if title else list(body_lines)
        document = "\n".join(blocks).strip() + "\n"
        return RenderedDocument(
            content=document.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            filename=f"{filename}.txt",
        )
