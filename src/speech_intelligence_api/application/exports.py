"""Turn a finished transcript into a downloadable transcript or summary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from speech_intelligence_api.application.summaries import split_sentences, summarize
from speech_intelligence_api.domain.enums import ExportContent, ExportFormat
from speech_intelligence_api.domain.errors import ErrorCode, ServiceError
from speech_intelligence_api.ports.documents import DocumentRenderer, RenderedDocument

_MAX_TRANSCRIPT_CHARS = 200_000
_FILENAME_STEM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ExportCommand:
    """A caller's request to download one view of a transcript."""

    transcript: str
    content: ExportContent
    export_format: ExportFormat
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.transcript.strip():
            raise ValueError("transcript cannot be empty")
        if len(self.transcript) > _MAX_TRANSCRIPT_CHARS:
            raise ValueError("transcript exceeds the exportable size limit")
        if self.title is not None and len(self.title) > 120:
            raise ValueError("title cannot exceed 120 characters")


class TranscriptExportService:
    """Build export documents without storing anything or calling a model."""

    def __init__(self, renderers: dict[ExportFormat, DocumentRenderer]) -> None:
        self._renderers = renderers

    def execute(self, command: ExportCommand) -> RenderedDocument:
        """Render the requested view; the transcript is never persisted."""

        renderer = self._renderers.get(command.export_format)
        if renderer is None:
            raise ServiceError(
                ErrorCode.INVALID_REQUEST,
                "The requested export format is not available.",
            )

        title = command.title or _default_title(command.content)
        return renderer.render(
            title=title,
            body_lines=self._body_lines(command),
            filename=_filename(title),
        )

    def _body_lines(self, command: ExportCommand) -> tuple[str, ...]:
        if command.content is ExportContent.TRANSCRIPT:
            return split_sentences(command.transcript) or (command.transcript.strip(),)

        key_points = summarize(command.transcript)
        if not key_points:
            # Too short to condense, so the transcript is already the summary.
            return split_sentences(command.transcript) or (command.transcript.strip(),)
        return tuple(f"• {point}" for point in key_points)


def _default_title(content: ExportContent) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    label = "Summary" if content is ExportContent.SUMMARY else "Transcript"
    return f"{label} {stamp}"


def _filename(title: str) -> str:
    """Build an ASCII download name; a native-script title stays in the document."""

    stem = _FILENAME_STEM.sub("-", title.casefold()).strip("-")
    return stem or "export"
