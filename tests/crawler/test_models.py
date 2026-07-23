from crawler.law_corpus.models import (
    AcquisitionSource,
    DocumentSpan,
    LegalUnit,
    SourceDocument,
)
from crawler.law_corpus.case_models import CaseDocument


def test_source_document_to_json_roundtrip():
    doc = SourceDocument(
        doc_id="china_pipl_2021",
        jurisdiction="CN",
        law_family="china_pipl",
        source_type="primary_law",
        title="中华人民共和国个人信息保护法",
        version_date="2021-08-20",
        effective_date="2021-11-01",
        source_url="https://example.test/pipl",
        language="zh",
        raw_text="第一条 为了保护个人信息权益...",
    )

    loaded = SourceDocument.from_json(doc.to_json())

    assert loaded.doc_id == "china_pipl_2021"
    assert loaded.raw_text.startswith("第一条")


def test_source_document_from_json_accepts_enriched_corpus_fields():
    loaded = SourceDocument.from_json(
        '{"doc_id":"eu_gdpr_2016_679","jurisdiction":"EU","law_family":"eu_gdpr",'
        '"source_type":"primary_law","title":"GDPR","version_date":"2016-04-27",'
        '"effective_date":"2018-05-25","source_url":"https://example.test",'
        '"language":"en","raw_text":"Article 1 text","raw_sha256":"abc123",'
        '"metadata":{"source_id":"eu_gdpr_2016_679"}}'
    )

    assert loaded.raw_sha256 == "abc123"
    assert loaded.metadata["source_id"] == "eu_gdpr_2016_679"


def test_case_document_from_json_accepts_normalized_raw_text():
    loaded = CaseDocument.from_json(
        '{"case_id":"gdprhub:3448","source_type":"case","title":"EDPB - Twitter",'
        '"source_url":"https://gdprhub.eu/example","language":"en",'
        '"raw_html":"<h2>Facts</h2>","raw_text":"Facts text",'
        '"categories":["Article_33_GDPR"],"external_links":[],'
        '"metadata":{"pageid":3448}}'
    )

    assert loaded.source_type == "case"
    assert loaded.raw_text == "Facts text"
    assert loaded.metadata["pageid"] == 3448


def test_legal_unit_keeps_citation_and_span_ids():
    unit = LegalUnit(
        unit_id="china_pipl_2021:article_1",
        source_doc_id="china_pipl_2021",
        parent_id=None,
        jurisdiction="CN",
        law_name="中华人民共和国个人信息保护法",
        version="2021-08-20",
        unit_type="article",
        canonical_citation="PIPL Article 1",
        local_citation="第一条",
        text="第一条 为了保护个人信息权益...",
        span_ids=["span-1"],
        parser_confidence=0.95,
        effective_from="2021-11-01",
        effective_to=None,
        is_current=True,
    )

    assert unit.span_ids == ["span-1"]
    assert unit.local_citation == "第一条"


def test_legal_unit_to_json_roundtrip():
    unit = LegalUnit(
        unit_id="eu_gdpr_2016_679:article_5",
        source_doc_id="eu_gdpr_2016_679",
        parent_id=None,
        jurisdiction="EU",
        law_name="GDPR",
        version="2016-04-27",
        unit_type="article",
        canonical_citation="GDPR Article 5",
        local_citation="Article 5",
        text="Article 5\nPersonal data shall be processed lawfully.",
        span_ids=["span-1"],
        parser_confidence=0.9,
        effective_from="2018-05-25",
        effective_to=None,
        is_current=True,
    )

    assert LegalUnit.from_json(unit.to_json()) == unit


def test_acquisition_source_exposes_manual_requirement():
    source = AcquisitionSource(
        doc_id="hong_kong_pdpo_cap486",
        title="Personal Data (Privacy) Ordinance Cap. 486",
        jurisdiction="HK",
        law_family="hong_kong_pdpo",
        source_type="primary_law",
        version_date="current",
        effective_date="",
        language="en-zh",
        url="https://www.elegislation.gov.hk/hk/cap486",
        preferred_format="txt",
        download_mode="manual",
        target_path="data/raw/laws/HK/hong_kong_pdpo/current/hong_kong_pdpo_cap486.txt",
        manual_instructions="Download and extract plain text.",
    )

    assert source.requires_manual_fetch is True


def test_document_span_tracks_offsets():
    span = DocumentSpan(
        span_id="span-1",
        source_doc_id="eu_gdpr_2016_679",
        text="Article 5 Principles relating to processing of personal data",
        char_start=100,
        char_end=160,
        heading="Article 5",
        section_path=["Article 5"],
        language="en",
    )

    assert span.char_end - span.char_start == 60
