import pytest

from crawler.law_corpus.models import SourceDocument
from crawler.law_corpus.parsers.base import make_span, stable_id


def test_stable_id_is_deterministic():
    assert stable_id("doc", "Article 1") == stable_id("doc", "Article 1")


def test_make_span_tracks_offsets():
    raw_text = "Prefix Article 1\nThis Regulation protects people."
    doc = SourceDocument(
        doc_id="eu_gdpr_2016_679",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text=raw_text,
    )

    char_start = len("Prefix ")
    span = make_span(doc, "Article 1", char_start, "Article 1", ["Article 1"])

    assert span.source_doc_id == "eu_gdpr_2016_679"
    assert span.char_start == char_start
    assert span.char_end == char_start + len("Article 1")


def test_make_span_rejects_misaligned_offsets():
    doc = SourceDocument(
        doc_id="eu_gdpr_2016_679",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text="Prefix Article 1\nThis Regulation protects people.",
    )

    with pytest.raises(ValueError, match="does not match source document text"):
        make_span(doc, "Article 1", 0, "Article 1", ["Article 1"])
