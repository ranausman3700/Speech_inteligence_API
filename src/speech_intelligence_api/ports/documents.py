"""Document-rendering boundary for transcript exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RenderedDocument:
    """An export ready to stream back to the caller."""

    content: bytes
    media_type: str
    filename: str


class DocumentRenderer(Protocol):
    """Turn a titled block of transcript text into a downloadable document."""

    def render(self, *, title: str, body_lines: tuple[str, ...], filename: str) -> RenderedDocument:
        """Return the encoded document; implementations never persist anything."""
