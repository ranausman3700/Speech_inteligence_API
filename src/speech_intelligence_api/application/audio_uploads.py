"""Shared bounded audio-upload validation for batch use cases."""

from __future__ import annotations

from pathlib import PurePath

from speech_intelligence_api.domain.errors import UnsupportedMediaTypeError

DECLARED_MEDIA_BY_EXTENSION: dict[str, frozenset[str]] = {
    ".aac": frozenset({"audio/aac"}),
    ".flac": frozenset({"audio/flac", "audio/x-flac"}),
    ".m4a": frozenset({"audio/mp4", "audio/x-m4a"}),
    ".mp3": frozenset({"audio/mpeg", "audio/mp3"}),
    ".oga": frozenset({"audio/ogg"}),
    ".ogg": frozenset({"audio/ogg"}),
    ".opus": frozenset({"audio/ogg", "audio/opus"}),
    ".wav": frozenset({"audio/wav", "audio/wave", "audio/x-wav"}),
    ".webm": frozenset({"audio/webm"}),
}

DETECTED_MEDIA_BY_EXTENSION: dict[str, frozenset[str]] = {
    ".aac": frozenset({"audio/aac"}),
    ".flac": frozenset({"audio/flac"}),
    ".m4a": frozenset({"audio/mp4"}),
    ".mp3": frozenset({"audio/mpeg"}),
    ".oga": frozenset({"audio/ogg"}),
    ".ogg": frozenset({"audio/ogg"}),
    ".opus": frozenset({"audio/ogg"}),
    ".wav": frozenset({"audio/wav"}),
    ".webm": frozenset({"audio/webm"}),
}


def validate_upload_identity(filename: str, declared_media_type: str) -> tuple[str, str]:
    """Return a trusted extension/media pair or reject the upload."""

    if not filename or len(filename) > 255 or "\x00" in filename:
        raise UnsupportedMediaTypeError
    normalized_filename = filename.replace("\\", "/")
    extension = PurePath(normalized_filename).suffix.casefold()
    media_type = declared_media_type.partition(";")[0].strip().casefold()
    accepted_media_types = DECLARED_MEDIA_BY_EXTENSION.get(extension)
    if accepted_media_types is None or media_type not in accepted_media_types:
        raise UnsupportedMediaTypeError
    return extension, media_type
