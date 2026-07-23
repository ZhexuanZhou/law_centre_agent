from __future__ import annotations

import sqlite3

from legal_agentic_retrieval.index import (
    RetrievalIndex,
    document_vector_path_for,
    passage_vector_path_for,
)
from legal_agentic_retrieval.models import ExactCitation, RetrievalFilters


def test_index_is_compact_and_excludes_non_current_records(built_index):
    path, _ = built_index

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(laws)")}
        unit_count = connection.execute("SELECT count(*) FROM law_units").fetchone()[0]
        vector_count = connection.execute("SELECT count(*) FROM vector_documents").fetchone()[0]
        passage_count = connection.execute("SELECT count(*) FROM passages").fetchone()[0]

    assert "raw_text" not in columns
    assert unit_count == 2
    assert vector_count == 3
    assert passage_count > 0
    assert document_vector_path_for(path).is_file()
    assert passage_vector_path_for(path).is_file()


def test_exact_retrieval_uses_structured_doc_id_and_citation(built_index):
    path, embedder = built_index
    index = RetrievalIndex(path, embedder)

    evidence = index.exact([ExactCitation(doc_id="eu_gdpr_2016_679", local_citation="ARTICLE 6")])

    assert [item.metadata["unit_id"] for item in evidence] == ["gdpr:article_6"]


def test_hydration_restores_complete_law_text(built_index):
    path, embedder = built_index
    index = RetrievalIndex(path, embedder)
    preview = index.exact([ExactCitation(doc_id="eu_gdpr_2016_679", local_citation="Article 6")])[0]

    hydrated = index.hydrate([preview])[0]

    assert preview.text.endswith("…")
    assert len(hydrated.text) > len(preview.text)
    assert hydrated.text.endswith("legal bases.")
    assert hydrated.metadata["passages"]


def test_vector_search_filters_source_type_without_domain_rules(built_index):
    path, embedder = built_index
    index = RetrievalIndex(path, embedder)

    evidence = index.vector_search(
        ["marketing consent"],
        RetrievalFilters(source_types=("case",), countries=("Spain",)),
        limit=5,
    )

    assert [item.evidence_id for item in evidence] == ["case:gdprhub:1"]
    assert evidence[0].matched_passage_ids


def test_catalog_exposes_data_values_for_model_planning(built_index):
    path, embedder = built_index

    catalog = RetrievalIndex(path, embedder).catalog()

    assert {item["doc_id"] for item in catalog["laws"]} == {
        "china_pipl_2021",
        "eu_gdpr_2016_679",
    }
    assert catalog["available_case_countries"] == ["Spain"]
    assert catalog["available_case_jurisdictions"] == ["ES"]
    gdpr = next(item for item in catalog["laws"] if item["doc_id"] == "eu_gdpr_2016_679")
    assert gdpr["citation_examples"]["article"] == "Article 6"
