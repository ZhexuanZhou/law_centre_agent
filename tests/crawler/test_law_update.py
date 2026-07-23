from __future__ import annotations

import json
from pathlib import Path

import pytest

from crawler.law_corpus.corpus_store import file_manifest, write_jsonl
from crawler.law_corpus.law_update import add_laws, validate_corpus
from crawler.law_corpus.models import LegalUnit


def _write_base_corpus(path: Path) -> None:
    path.mkdir()
    law = {
        "doc_id": "existing_law_2025",
        "title": "Existing Law",
        "jurisdiction": "EU",
        "law_family": "eu_gdpr",
        "source_type": "primary_law",
        "raw_text": "Article 1 Existing provision.",
    }
    unit = LegalUnit(
        unit_id="existing_law_2025:article_1",
        source_doc_id="existing_law_2025",
        parent_id=None,
        jurisdiction="EU",
        law_name="Existing Law",
        version="2025-01-01",
        unit_type="article",
        canonical_citation="Existing Law Article 1",
        local_citation="Article 1",
        text="Article 1 Existing provision.",
        span_ids=["existing_law_2025:span:1"],
        parser_confidence=1.0,
        effective_from="2025-01-01",
        effective_to=None,
        is_current=True,
    )
    case = {
        "case_id": "case:existing",
        "title": "Existing Case",
        "relevant_laws": [],
    }
    records = {
        "laws.jsonl": [law],
        "legal_units.jsonl": [unit],
        "law_relations.jsonl": [],
        "gdprhub_cases.jsonl": [case],
        "case_law_relations.jsonl": [],
    }
    entries = []
    for filename, values in records.items():
        count = write_jsonl(values, path / filename)
        entries.append(file_manifest(path / filename, record_count=count))
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": "2.0.0", "files": entries}),
        encoding="utf-8",
    )


def _write_new_law_catalog(path: Path, raw_path: Path, *, doc_id: str) -> None:
    path.write_text(
        f'''[[sources]]
doc_id = "{doc_id}"
title = "测试个人信息法"
jurisdiction = "CN"
law_family = "china_pipl"
source_type = "primary_law"
version_date = "2026-01-01"
effective_date = "2026-02-01"
language = "zh"
url = "https://example.test/law"
preferred_format = "txt"
download_mode = "manual"
target_path = "{raw_path}"
source_set = "custom"
translation_status = "official_original"
''',
        encoding="utf-8",
    )


def test_add_laws_stages_new_corpus_and_preserves_cases(tmp_path: Path) -> None:
    base = tmp_path / "structured"
    output = tmp_path / "structured_candidate"
    raw = tmp_path / "new_law.txt"
    catalog = tmp_path / "new_law.toml"
    _write_base_corpus(base)
    raw.write_text(
        "第一条\n为了保护个人信息权益，制定本法。\n第二条\n个人信息受法律保护。\n",
        encoding="utf-8",
    )
    _write_new_law_catalog(catalog, raw, doc_id="test_personal_information_law_2026")

    report = add_laws(
        catalog_paths=[catalog],
        base_corpus_dir=base,
        output_corpus_dir=output,
        acquire=False,
    )

    assert report["new_doc_ids"] == ["test_personal_information_law_2026"]
    assert report["new_legal_unit_count"] == 2
    assert report["record_counts"]["laws.jsonl"] == 2
    assert report["record_counts"]["gdprhub_cases.jsonl"] == 1
    assert validate_corpus(output)["valid"] is True
    assert validate_corpus(base)["valid"] is True

    update_report = output / "update_report.json"
    update_report.write_text(update_report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    corrupted = validate_corpus(output)
    assert corrupted["valid"] is False
    assert "Manifest SHA-256 mismatch for update_report.json" in corrupted["errors"]


def test_add_laws_rejects_existing_doc_id_and_keeps_base_unchanged(tmp_path: Path) -> None:
    base = tmp_path / "structured"
    output = tmp_path / "structured_candidate"
    raw = tmp_path / "new_law.txt"
    catalog = tmp_path / "new_law.toml"
    _write_base_corpus(base)
    raw.write_text("第一条\n重复法规。\n", encoding="utf-8")
    _write_new_law_catalog(catalog, raw, doc_id="existing_law_2025")

    with pytest.raises(ValueError, match="already exists"):
        add_laws(
            catalog_paths=[catalog],
            base_corpus_dir=base,
            output_corpus_dir=output,
            acquire=False,
        )

    assert not output.exists()
    assert validate_corpus(base)["valid"] is True


def test_add_laws_rejects_in_place_output(tmp_path: Path) -> None:
    base = tmp_path / "structured"
    raw = tmp_path / "new_law.txt"
    catalog = tmp_path / "new_law.toml"
    _write_base_corpus(base)
    raw.write_text("第一条\n新增法规。\n", encoding="utf-8")
    _write_new_law_catalog(catalog, raw, doc_id="new_law_2026")

    with pytest.raises(ValueError, match="in-place updates are disabled"):
        add_laws(
            catalog_paths=[catalog],
            base_corpus_dir=base,
            output_corpus_dir=base,
            acquire=False,
        )
