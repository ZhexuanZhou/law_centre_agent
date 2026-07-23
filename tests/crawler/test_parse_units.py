import json
import importlib.util
import subprocess
import sys
from pathlib import Path

from crawler.law_corpus.models import LegalUnit, SourceDocument
from crawler.law_corpus.parse_units import (
    DuplicateLegalUnitError,
    ParseCoverageError,
    dedupe_legal_units,
    parse_source_documents,
    read_legal_units_jsonl,
    read_source_documents_jsonl,
    read_source_documents_jsonl_many,
    write_legal_units_jsonl,
)
from crawler.law_corpus.parsers.registry import get_parser


def _load_script_module(name: str, script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, repo_root / "tools" / script_name)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def gdpr_source_document(doc_id: str = "eu_gdpr_2016_679") -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text="Article 5\nPrinciples\n1. Personal data shall be processed lawfully.",
    )


def test_registry_covers_supported_law_families():
    for law_family in [
        "eu_gdpr",
        "eu_ai_act",
        "china_pipl",
        "china_dsl",
        "us_cfr",
        "us_state_privacy",
        "singapore_pdpa",
        "hong_kong_pdpo",
        "india_dpdp",
        "eu_data_act",
        "eu_data_governance_act",
        "eu_dsa",
        "eu_nis2",
        "china_csl",
        "uk_dpa",
        "canada_pipeda",
        "australia_privacy_act",
        "japan_appi",
        "korea_pipa",
        "brazil_lgpd",
        "south_africa_popia",
        "philippines_dpa",
        "malaysia_pdpa",
    ]:
        assert get_parser(law_family).source_families


def test_parse_legal_units_cli_defaults_to_full_corpus_paths():
    module = _load_script_module("parse_legal_units_script", "parse_legal_units.py")

    assert module.DEFAULT_SOURCE_DOCUMENTS == "corpus/normalized/source_documents.full.jsonl"
    assert module.DEFAULT_OUT == "corpus/parsed/legal_units.full.jsonl"


def test_parse_source_documents_uses_registry_and_writes_jsonl(tmp_path: Path):
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
        raw_text="Article 5\nPrinciples\n1. Personal data shall be processed lawfully.",
    )

    units = parse_source_documents([doc], require_all=True)

    assert len(units) == 2
    assert units[0].canonical_citation == "GDPR Article 5"
    assert units[1].canonical_citation == "GDPR Article 5(1)"

    source_path = tmp_path / "source_documents.jsonl"
    source_path.write_text(doc.to_json() + "\n", encoding="utf-8")
    assert read_source_documents_jsonl(source_path)[0].doc_id == "eu_gdpr_2016_679"

    out = tmp_path / "legal_units.jsonl"
    write_legal_units_jsonl(units, out)
    assert out.read_text(encoding="utf-8").count("\n") == 2
    assert read_legal_units_jsonl(out)[0].source_doc_id == "eu_gdpr_2016_679"
    payload = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert payload["source_doc_id"] == "eu_gdpr_2016_679"


def test_parse_source_documents_can_require_every_doc_to_parse():
    doc = SourceDocument(
        doc_id="empty_doc",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text="No article headings here.",
    )

    try:
        parse_source_documents([doc], require_all=True)
    except ParseCoverageError as exc:
        assert "empty_doc" in str(exc)
    else:
        raise AssertionError("Expected ParseCoverageError")


def test_dedupe_legal_units_keeps_identical_duplicate_unit_id():
    unit = parse_source_documents(
        [
            SourceDocument(
                doc_id="eu_gdpr_2016_679",
                jurisdiction="EU",
                law_family="eu_gdpr",
                source_type="primary_law",
                title="GDPR",
                version_date="2016-04-27",
                effective_date="2018-05-25",
                source_url="https://example.test",
                language="en",
                raw_text="Article 5\nPrinciples\n1. Personal data shall be processed lawfully.",
            )
        ],
        require_all=True,
    )[0]

    assert dedupe_legal_units([unit, unit]) == [unit]


def test_dedupe_legal_units_rejects_conflicting_duplicate_unit_id():
    unit = parse_source_documents(
        [
            SourceDocument(
                doc_id="eu_gdpr_2016_679",
                jurisdiction="EU",
                law_family="eu_gdpr",
                source_type="primary_law",
                title="GDPR",
                version_date="2016-04-27",
                effective_date="2018-05-25",
                source_url="https://example.test",
                language="en",
                raw_text="Article 5\nPrinciples\n1. Personal data shall be processed lawfully.",
            )
        ],
        require_all=True,
    )[0]
    changed = LegalUnit(**{**unit.__dict__, "text": "Different text"})

    try:
        dedupe_legal_units([unit, changed])
    except DuplicateLegalUnitError as exc:
        message = str(exc)
        assert unit.unit_id in message
        assert "differing_fields=text" in message
        assert "existing(" in message
        assert "incoming(" in message
        assert "source_doc_id=eu_gdpr_2016_679" in message
        assert "canonical_citation=GDPR Article 5" in message
        assert "unit_type=article" in message
        assert "local_citation=Article 5" in message
        assert "text_len=" in message
        assert "text_sha256=" in message
    else:
        raise AssertionError("Expected DuplicateLegalUnitError")


def test_read_source_documents_jsonl_many_preserves_cross_file_order(tmp_path: Path):
    first = gdpr_source_document("first_doc")
    second = gdpr_source_document("second_doc")
    third = gdpr_source_document("third_doc")
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_text(
        first.to_json() + "\n" + second.to_json() + "\n",
        encoding="utf-8",
    )
    second_path.write_text(third.to_json() + "\n", encoding="utf-8")

    documents = read_source_documents_jsonl_many([first_path, second_path])

    assert [document.doc_id for document in documents] == [
        "first_doc",
        "second_doc",
        "third_doc",
    ]


def test_parse_legal_units_cli_accepts_repeated_source_documents_and_dedupes(
    tmp_path: Path,
):
    document = gdpr_source_document()
    first_path = tmp_path / "first_source_documents.jsonl"
    second_path = tmp_path / "second_source_documents.jsonl"
    out_path = tmp_path / "legal_units.jsonl"
    first_path.write_text(document.to_json() + "\n", encoding="utf-8")
    second_path.write_text(document.to_json() + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "parse_legal_units.py"),
            "--source-documents",
            str(first_path),
            "--source-documents",
            str(second_path),
            "--out",
            str(out_path),
            "--require-all",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    units = read_legal_units_jsonl(out_path)
    assert len(units) == 2
    assert [unit.canonical_citation for unit in units] == [
        "GDPR Article 5",
        "GDPR Article 5(1)",
    ]
    assert "wrote 2 legal units" in result.stdout


def test_parse_legal_units_cli_requires_out_for_noncanonical_input(tmp_path: Path):
    document = gdpr_source_document()
    source_path = tmp_path / "source_documents.jsonl"
    source_path.write_text(document.to_json() + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "parse_legal_units.py"),
            "--source-documents",
            str(source_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--out is required" in result.stderr
