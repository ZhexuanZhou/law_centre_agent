from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path

from crawler.law_corpus.models import LegalUnit, SourceDocument
from crawler.law_corpus.parsers.registry import get_parser


class ParseCoverageError(ValueError):
    pass


class DuplicateLegalUnitError(ValueError):
    pass


def _differing_legal_unit_fields(existing: LegalUnit, incoming: LegalUnit) -> list[str]:
    return [
        field.name
        for field in fields(LegalUnit)
        if getattr(existing, field.name) != getattr(incoming, field.name)
    ]


def _text_sha256_prefix(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _legal_unit_summary(unit: LegalUnit) -> str:
    return (
        f"source_doc_id={unit.source_doc_id}, "
        f"canonical_citation={unit.canonical_citation}, "
        f"unit_type={unit.unit_type}, "
        f"local_citation={unit.local_citation}, "
        f"text_len={len(unit.text)}, "
        f"text_sha256={_text_sha256_prefix(unit.text)}"
    )


def dedupe_legal_units(units: list[LegalUnit]) -> list[LegalUnit]:
    by_unit_id: dict[str, LegalUnit] = {}
    ordered: list[LegalUnit] = []
    for unit in units:
        existing = by_unit_id.get(unit.unit_id)
        if existing is None:
            by_unit_id[unit.unit_id] = unit
            ordered.append(unit)
            continue
        if existing != unit:
            differing_fields = _differing_legal_unit_fields(existing, unit)
            raise DuplicateLegalUnitError(
                f"Conflicting LegalUnit records for unit_id={unit.unit_id}; "
                f"differing_fields={', '.join(differing_fields)}; "
                f"existing({_legal_unit_summary(existing)}); "
                f"incoming({_legal_unit_summary(unit)})"
            )
    return ordered


def parse_source_documents(
    documents: list[SourceDocument],
    *,
    require_all: bool = False,
) -> list[LegalUnit]:
    units: list[LegalUnit] = []
    empty_doc_ids: list[str] = []
    for document in documents:
        parser = get_parser(document.law_family)
        document_units = parser.parse(document)
        if require_all and not document_units:
            empty_doc_ids.append(document.doc_id)
        units.extend(document_units)

    if empty_doc_ids:
        raise ParseCoverageError(
            "No LegalUnit parsed for source documents: " + ", ".join(empty_doc_ids)
        )
    return dedupe_legal_units(units)


def read_source_documents_jsonl(path: str | Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(SourceDocument.from_json(line))
    return documents


def read_source_documents_jsonl_many(paths: list[str | Path]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in paths:
        documents.extend(read_source_documents_jsonl(path))
    return documents


def read_legal_units_jsonl(path: str | Path) -> list[LegalUnit]:
    units: list[LegalUnit] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                units.append(LegalUnit.from_json(line))
    return units


def write_legal_units_jsonl(units: list[LegalUnit], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(unit.to_json())
            handle.write("\n")
