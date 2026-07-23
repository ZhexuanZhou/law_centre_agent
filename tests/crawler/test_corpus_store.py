import json

from crawler.law_corpus.corpus_store import (
    build_case_law_relations,
    build_law_relations,
    scan_case_candidates,
    structure_gdprhub_case,
)
from crawler.law_corpus.models import LegalUnit


def _legal_unit(
    unit_id: str,
    source_doc_id: str,
    text: str,
    *,
    local_citation: str = "Article 1",
    parent_id: str | None = None,
    unit_type: str = "article",
    law_name: str | None = None,
) -> LegalUnit:
    return LegalUnit(
        unit_id=unit_id,
        source_doc_id=source_doc_id,
        parent_id=parent_id,
        jurisdiction="EU",
        law_name=law_name or source_doc_id,
        version="current",
        unit_type=unit_type,
        canonical_citation=f"{source_doc_id} {local_citation}",
        local_citation=local_citation,
        text=text,
        span_ids=[f"{unit_id}:span"],
        parser_confidence=0.9,
        effective_from=None,
        effective_to=None,
        is_current=True,
    )


def test_scan_case_candidates_dedupes_and_selects_latest_html(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {
            "case_id": "gdprhub:1",
            "title": "Authority - Case",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "raw_html": "old",
            "metadata": {"requested_title": "Old alias"},
        },
        {
            "case_id": "gdprhub:1",
            "title": "Authority - Case",
            "retrieved_at": "2026-02-01T00:00:00Z",
            "raw_html": "new and longer",
            "metadata": {"requested_title": "New alias"},
        },
        {
            "case_id": "gdprhub:2",
            "title": "Authority - Other",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "raw_html": "other",
            "metadata": {},
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selection = scan_case_candidates(path)

    assert selection.input_rows == 3
    assert selection.duplicate_rows == 1
    assert selection.duplicate_case_ids == 1
    assert selection.selected_line_by_case_id["gdprhub:1"] == 2
    assert selection.requested_titles_by_case_id["gdprhub:1"] == [
        "Old alias",
        "New alias",
    ]
    assert selection.conflicting_content_case_ids == ["gdprhub:1"]


def test_structure_gdprhub_case_extracts_required_fields_and_resolves_gdpr():
    raw_html = (
        "<div class='mw-parser-output'>"
        "<table class='wikitable'>"
        "<tr><th>Authority:</th><td>AEPD (Spain)</td></tr>"
        "<tr><th>Jurisdiction:</th><td>Spain</td></tr>"
        "<tr><th>Relevant Law:</th><td><a>Article 6(1)(a) GDPR</a></td></tr>"
        "<tr><th>Parties:</th><td>Example Retail Bank</td></tr>"
        "<tr><th>Outcome:</th><td>Violation Found</td></tr>"
        "<tr><th>Decided:</th><td>02.01.2025</td></tr>"
        "</table>"
        "<h2>English Summary</h2>"
        "<h3>Facts</h3><p>A retail bank sent marketing emails.</p>"
        "<h3>Holding</h3><p>The authority found a GDPR violation.</p>"
        "</div>"
    )
    payload = {
        "case_id": "gdprhub:1",
        "source_type": "case",
        "title": "AEPD (Spain) - PS/1/2025",
        "source_url": "https://gdprhub.eu/example",
        "language": "en",
        "raw_html": raw_html,
        "raw_text": "",
        "categories": ["Article_6(1)(a)_GDPR", "Spain"],
        "external_links": [],
        "retrieved_at": "2026-01-01T00:00:00Z",
        "metadata": {},
    }

    record = structure_gdprhub_case(
        payload,
        legal_unit_index={
            ("eu_gdpr_2016_679", "article 6(1)(a)"): "gdpr:article_6:paragraph_1:point_a"
        },
    )

    assert record["jurisdiction"] == "ES"
    assert record["country"] == "Spain"
    assert record["company_or_parties"] == "Example Retail Bank"
    assert {item["industry"] for item in record["industry"]} >= {
        "finance_insurance",
        "retail_ecommerce",
    }
    assert record["facts_text"] == "A retail bank sent marketing emails."
    assert "GDPR violation" in record["decision_text"]
    assert record["relevant_laws"][0]["target_unit_id"] == ("gdpr:article_6:paragraph_1:point_a")
    assert record["decided_date"] == "2025-01-02"
    assert record["extraction_quality"]["status"] == "complete"


def test_build_law_relations_resolves_intra_and_cross_law_units():
    units = [
        _legal_unit(
            "data-act:article_1",
            "eu_data_act_2023_2854",
            "This Regulation complements Regulation (EU) 2016/679.",
        ),
        _legal_unit(
            "data-act:article_2",
            "eu_data_act_2023_2854",
            "Personal data has the meaning given in Article 4, point (1), of Regulation (EU) 2016/679.",
            local_citation="Article 2",
        ),
        _legal_unit(
            "gdpr:article_4:paragraph_1",
            "eu_gdpr_2016_679",
            "Personal data means information relating to an identified person.",
            local_citation="Article 4(1)",
            unit_type="paragraph",
            law_name="General Data Protection Regulation",
        ),
        _legal_unit(
            "gdpr:article_5",
            "eu_gdpr_2016_679",
            "Processing shall comply with Article 6(1), and Articles 12 to 14 apply.",
            local_citation="Article 5",
            law_name="General Data Protection Regulation",
        ),
        *[
            _legal_unit(
                f"gdpr:article_{citation.lower().replace('(', '_').replace(')', '')}",
                "eu_gdpr_2016_679",
                "Target provision.",
                local_citation=f"Article {citation}",
                law_name="General Data Protection Regulation",
            )
            for citation in ("6(1)", "12", "13", "14")
        ],
    ]

    relations = build_law_relations(units)

    assert len(relations) == 5
    cross_law = next(item for item in relations if item["relation_scope"] == "cross_law")
    assert cross_law["source_unit_id"] == "data-act:article_2"
    assert cross_law["target_unit_id"] == "gdpr:article_4:paragraph_1"
    assert cross_law["target_requested_citation"] == "Article 4(1)"
    assert cross_law["matched_law_aliases"] == ["Regulation (EU) 2016/679"]
    assert {
        item["target_requested_citation"]
        for item in relations
        if item["relation_scope"] == "intra_law"
    } == {"Article 6(1)", "Article 12", "Article 13", "Article 14"}
    assert all(item["relation_type"] == "cites" for item in relations)
    assert all(item["source_unit_id"] != item["target_unit_id"] for item in relations)


def test_build_law_relations_resolves_section_subdivision_to_parent_unit():
    units = [
        _legal_unit(
            "uk:section_1",
            "uk_data_protection_act_2018",
            "See sections 2 and 205(4).",
            local_citation="Section 1",
            unit_type="section",
            law_name="Data Protection Act 2018",
        ),
        _legal_unit(
            "uk:section_2",
            "uk_data_protection_act_2018",
            "Target section.",
            local_citation="Section 2",
            unit_type="section",
            law_name="Data Protection Act 2018",
        ),
        _legal_unit(
            "uk:section_205",
            "uk_data_protection_act_2018",
            "Target section with subsections.",
            local_citation="Section 205",
            unit_type="section",
            law_name="Data Protection Act 2018",
        ),
    ]

    relations = build_law_relations(units)

    assert {item["target_requested_citation"] for item in relations} == {
        "Section 2",
        "Section 205(4)",
    }
    fallback = next(
        item for item in relations if item["target_requested_citation"] == "Section 205(4)"
    )
    assert fallback["target_unit_id"] == "uk:section_205"
    assert fallback["resolution_basis"] == "parent_unit_fallback"
    assert fallback["confidence"] == 0.95


def test_build_law_relations_does_not_infer_relation_from_document_name_only():
    units = [
        _legal_unit(
            "data-act:article_1",
            "eu_data_act_2023_2854",
            "This Regulation complements Regulation (EU) 2016/679.",
        ),
        _legal_unit(
            "gdpr:article_1",
            "eu_gdpr_2016_679",
            "This Regulation protects natural persons.",
        ),
    ]

    assert build_law_relations(units) == []


def test_build_law_relations_does_not_map_unknown_external_instrument_to_source_law():
    units = [
        _legal_unit(
            "ai-act:article_1",
            "eu_ai_act_2024_1689",
            "Article 10 of Directive (EU) 2016/680 applies.",
            local_citation="Article 1",
            law_name="EU AI Act",
        ),
        _legal_unit(
            "ai-act:article_10",
            "eu_ai_act_2024_1689",
            "Target provision in this Regulation.",
            local_citation="Article 10",
            law_name="EU AI Act",
        ),
    ]

    assert build_law_relations(units) == []


def test_build_law_relations_keeps_adjacent_external_instruments_separate():
    units = [
        _legal_unit(
            "ai-act:recital_1",
            "eu_ai_act_2024_1689",
            (
                "Biometric data is defined in Article 4, point (14), of Regulation (EU) "
                "2016/679, and Article 3, point (18), of Regulation (EU) 2018/1725."
            ),
            local_citation="Recital 1",
            unit_type="recital",
            law_name="EU AI Act",
        ),
        _legal_unit(
            "ai-act:article_3",
            "eu_ai_act_2024_1689",
            "Definitions in this Regulation.",
            local_citation="Article 3",
            law_name="EU AI Act",
        ),
        _legal_unit(
            "gdpr:article_4:paragraph_14",
            "eu_gdpr_2016_679",
            "Biometric data definition.",
            local_citation="Article 4(14)",
            unit_type="paragraph",
            law_name="General Data Protection Regulation",
        ),
    ]

    relations = build_law_relations(units)

    assert len(relations) == 1
    assert relations[0]["target_unit_id"] == "gdpr:article_4:paragraph_14"
    assert relations[0]["target_requested_citation"] == "Article 4(14)"


def test_build_case_law_relations_preserves_resolved_and_unresolved_references():
    units = [
        _legal_unit(
            "gdpr:article_6:paragraph_1:point_a",
            "eu_gdpr_2016_679",
            "Consent is a lawful basis.",
            local_citation="Article 6(1)(a)",
            unit_type="point",
            law_name="General Data Protection Regulation",
        )
    ]
    cases = [
        {
            "case_id": "gdprhub:1",
            "title": "AEPD - Example",
            "relevant_laws": [
                {
                    "citation": "Article 6(1)(a) GDPR",
                    "instrument": "GDPR",
                    "target_doc_id": "eu_gdpr_2016_679",
                    "target_unit_id": "gdpr:article_6:paragraph_1:point_a",
                    "resolution_status": "resolved",
                },
                {
                    "citation": "Article 22 PECR",
                    "instrument": None,
                    "target_doc_id": None,
                    "target_unit_id": None,
                    "resolution_status": "unresolved",
                },
            ],
        }
    ]

    relations = build_case_law_relations(cases, units)

    assert len(relations) == 2
    resolved = next(item for item in relations if item["resolution_status"] == "resolved")
    assert resolved["case_id"] == "gdprhub:1"
    assert resolved["target_unit_id"] == "gdpr:article_6:paragraph_1:point_a"
    assert resolved["target_citation"].endswith("Article 6(1)(a)")
    unresolved = next(item for item in relations if item["resolution_status"] == "unresolved")
    assert unresolved["target_unit_id"] is None
    assert unresolved["citation"] == "Article 22 PECR"
