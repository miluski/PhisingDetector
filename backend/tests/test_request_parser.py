"""Tests for the shared email and SMS request parser."""

from types import SimpleNamespace

import pytest

from app.request_parser import AnalysisRequestParser


class UploadedFile:
    """Small upload fake implementing the Flask file contract."""

    filename = "note.txt"

    def read(self):
        """Return deterministic attachment bytes."""
        return b"https://example.com"


def test_sms_parser_applies_defaults():
    """SMS fields are normalized and default values are stable."""
    request = AnalysisRequestParser().parse_sms(SimpleNamespace(get=lambda key: {"content": " hi "}.get(key)))
    assert request.content == "hi"
    assert request.models == ["random_forest"]
    assert request.mode == "optimized"
    assert request.dataset == "current"


def test_email_parser_reads_sender_receiver_and_files():
    """Email parser preserves address fields and uploaded bytes."""
    values = {"content": " body ", "sender": " a@example.com ", "receiver": " b@example.com "}
    request = AnalysisRequestParser().parse_email(SimpleNamespace(get=values.get), [UploadedFile()])
    assert request.sender == "a@example.com"
    assert request.receiver == "b@example.com"
    assert request.attachments == [{"filename": "note.txt", "content": b"https://example.com"}]


def test_empty_sms_body_is_rejected():
    """Empty SMS content returns the same validation error as before."""
    with pytest.raises(ValueError, match="content"):
        AnalysisRequestParser().parse_sms(SimpleNamespace(get=lambda _: " "))
