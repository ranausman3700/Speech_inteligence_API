"""Transcript export, summarization and document-renderer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_intelligence_api.adapters.pdf_documents import PdfDocumentRenderer
from speech_intelligence_api.adapters.text_documents import PlainTextDocumentRenderer
from speech_intelligence_api.application.exports import ExportCommand, TranscriptExportService
from speech_intelligence_api.application.summaries import split_sentences, summarize
from speech_intelligence_api.domain.enums import ExportContent, ExportFormat
from speech_intelligence_api.domain.errors import ServiceError
from speech_intelligence_api.ports.documents import DocumentRenderer
from tests.factories import make_settings

_CONVERSATION = (
    "So yeah so yeah. The migraine started on Tuesday evening. Yeah so yeah. "
    "I took ibuprofen before bed. Yeah okay. So yeah so."
)


def _service(**renderers: DocumentRenderer) -> TranscriptExportService:
    mapping: dict[ExportFormat, DocumentRenderer] = {
        ExportFormat[name.upper()]: renderer for name, renderer in renderers.items()
    }
    return TranscriptExportService(mapping)


def test_sentences_split_on_marks_from_every_supported_script() -> None:
    assert split_sentences("One. Two! Three?") == ("One.", "Two!", "Three?")
    assert split_sentences("你好。再见！") == ("你好。", "再见！")  # noqa: RUF001
    assert split_sentences("मैं ठीक हूं। नमस्ते।") == ("मैं ठीक हूं।", "नमस्ते।")


def test_short_transcripts_are_their_own_summary() -> None:
    assert summarize("One. Two. Three.") == ()


def test_summary_keeps_distinctive_sentences_in_spoken_order() -> None:
    assert summarize(_CONVERSATION) == (
        "The migraine started on Tuesday evening.",
        "I took ibuprofen before bed.",
    )


def test_summary_stays_glanceable_on_long_transcripts() -> None:
    transcript = " ".join(f"Distinct sentence number {n}." for n in range(60))

    assert len(summarize(transcript)) == 5


def test_summary_ignores_repeated_filler_in_scripts_without_spaces() -> None:
    filler = "今天天气很好。"
    transcript = filler + "我昨天开始头痛得很厉害。" + filler + "我吃了布洛芬。" + filler * 2

    key_points = summarize(transcript)

    assert key_points
    assert filler not in key_points


def test_plain_text_export_is_utf8_with_a_download_name() -> None:
    document = PlainTextDocumentRenderer().render(
        title="Summary",
        body_lines=("• مرحبا", "• 你好"),
        filename="summary",
    )

    assert document.filename == "summary.txt"
    assert document.media_type == "text/plain; charset=utf-8"
    assert "مرحبا" in document.content.decode("utf-8")


def test_transcript_export_splits_the_text_into_sentences() -> None:
    service = _service(txt=PlainTextDocumentRenderer())

    document = service.execute(
        ExportCommand(
            transcript="One. Two. Three.",
            content=ExportContent.TRANSCRIPT,
            export_format=ExportFormat.TXT,
        )
    )

    body = document.content.decode("utf-8")
    assert "One." in body and "Three." in body


def test_summary_export_renders_bullets_and_falls_back_when_too_short() -> None:
    service = _service(txt=PlainTextDocumentRenderer())

    summarized = service.execute(
        ExportCommand(
            transcript=_CONVERSATION,
            content=ExportContent.SUMMARY,
            export_format=ExportFormat.TXT,
        )
    )
    too_short = service.execute(
        ExportCommand(
            transcript="Only one sentence here.",
            content=ExportContent.SUMMARY,
            export_format=ExportFormat.TXT,
        )
    )

    assert "• The migraine started on Tuesday evening." in summarized.content.decode("utf-8")
    # Nothing to condense, so the caller still gets usable text rather than an empty file.
    assert "Only one sentence here." in too_short.content.decode("utf-8")


def test_a_custom_title_is_used_for_both_the_heading_and_the_filename() -> None:
    service = _service(txt=PlainTextDocumentRenderer())

    document = service.execute(
        ExportCommand(
            transcript="One. Two. Three.",
            content=ExportContent.TRANSCRIPT,
            export_format=ExportFormat.TXT,
            title="Visit Notes 2026",
        )
    )

    assert document.filename == "visit-notes-2026.txt"
    assert "Visit Notes 2026" in document.content.decode("utf-8")


def test_a_native_script_title_still_produces_an_ascii_download_name() -> None:
    service = _service(txt=PlainTextDocumentRenderer())

    document = service.execute(
        ExportCommand(
            transcript="One. Two.",
            content=ExportContent.TRANSCRIPT,
            export_format=ExportFormat.TXT,
            title="ملاحظات",
        )
    )

    assert document.filename == "export.txt"
    assert "ملاحظات" in document.content.decode("utf-8")


def test_an_unavailable_format_is_reported_rather_than_crashing() -> None:
    service = _service(txt=PlainTextDocumentRenderer())

    with pytest.raises(ServiceError):
        service.execute(
            ExportCommand(
                transcript="One. Two.",
                content=ExportContent.TRANSCRIPT,
                export_format=ExportFormat.PDF,
            )
        )


def test_blank_and_oversized_transcripts_are_refused() -> None:
    with pytest.raises(ValueError):
        ExportCommand(
            transcript="   ",
            content=ExportContent.TRANSCRIPT,
            export_format=ExportFormat.TXT,
        )
    with pytest.raises(ValueError):
        ExportCommand(
            transcript="a" * 200_001,
            content=ExportContent.TRANSCRIPT,
            export_format=ExportFormat.TXT,
        )


def _pdf_renderer(font_root: Path | None) -> PdfDocumentRenderer:
    return PdfDocumentRenderer(make_settings().model_copy(update={"export_font_root": font_root}))


def test_pdf_export_without_a_font_directory_tells_the_caller_to_use_text() -> None:
    with pytest.raises(ServiceError) as raised:
        _pdf_renderer(None).render(title="T", body_lines=("hello",), filename="t")

    assert "Export txt instead" in raised.value.public_message


def test_pdf_export_names_the_missing_font_for_an_uninstalled_script(tmp_path: Path) -> None:
    (tmp_path / "NotoSans-Regular.ttf").touch()

    with pytest.raises(ServiceError) as raised:
        _pdf_renderer(tmp_path).render(title="T", body_lines=("ฉันปวดหัว",), filename="t")

    assert "NotoSansThai" in raised.value.public_message
