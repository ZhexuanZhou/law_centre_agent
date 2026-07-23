from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from crawler.law_corpus.catalog import load_sources
from crawler.law_corpus.extract_text import (
    extract_text_from_file,
    extract_uk_legislation_metadata_from_file,
)
from crawler.law_corpus.models import AcquisitionSource, SourceDocument


class DuplicateSourceDocumentError(ValueError):
    pass


SUPERSEDED_SOURCE_DOCUMENTS = {
    "us_ccpa_cpra_1798_100": "us_ca_ccpa_cpra_civ_1798_100_199",
}


def build_source_document(source: AcquisitionSource) -> SourceDocument:
    raw_path = Path(source.target_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw law source is missing: {raw_path}")
    metadata: dict[str, object] = {
        "source_set": source.source_set,
        "translation_status": source.translation_status,
        "acquisition_mode": source.download_mode,
        "preferred_format": source.preferred_format,
        "target_path": source.target_path,
    }
    uk_legislation_metadata = extract_uk_legislation_metadata_from_file(raw_path)
    if uk_legislation_metadata:
        metadata["uk_legislation"] = uk_legislation_metadata
    return SourceDocument(
        doc_id=source.doc_id,
        jurisdiction=source.jurisdiction,
        law_family=source.law_family,
        source_type=source.source_type,
        title=source.title,
        version_date=source.version_date or None,
        effective_date=source.effective_date or None,
        source_url=source.url,
        language=source.language,
        raw_text=extract_text_from_file(raw_path),
        metadata=metadata,
    )


def build_source_documents_from_catalog(catalog_path: str | Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for source in load_sources(catalog_path):
        if Path(source.target_path).exists():
            documents.append(build_source_document(source))
    return documents


def _differing_source_document_fields(
    existing: SourceDocument, incoming: SourceDocument
) -> list[str]:
    return [
        field.name
        for field in fields(SourceDocument)
        if getattr(existing, field.name) != getattr(incoming, field.name)
    ]


def _source_document_summary(document: SourceDocument) -> str:
    return (
        f"metadata.source_set={document.metadata.get('source_set')}, "
        f"metadata.target_path={document.metadata.get('target_path')}, "
        f"source_url={document.source_url}"
    )


def dedupe_source_documents(documents: list[SourceDocument]) -> list[SourceDocument]:
    by_doc_id: dict[str, SourceDocument] = {}
    ordered: list[SourceDocument] = []
    for document in documents:
        existing = by_doc_id.get(document.doc_id)
        if existing is None:
            by_doc_id[document.doc_id] = document
            ordered.append(document)
            continue
        if existing != document:
            differing_fields = _differing_source_document_fields(existing, document)
            raise DuplicateSourceDocumentError(
                f"Conflicting SourceDocument records for doc_id={document.doc_id}; "
                f"differing_fields={', '.join(differing_fields)}; "
                f"existing({_source_document_summary(existing)}); "
                f"incoming({_source_document_summary(document)})"
            )
    return ordered


def drop_superseded_source_documents(documents: list[SourceDocument]) -> list[SourceDocument]:
    doc_ids = {document.doc_id for document in documents}
    return [
        document
        for document in documents
        if SUPERSEDED_SOURCE_DOCUMENTS.get(document.doc_id) not in doc_ids
    ]


def build_source_documents_from_catalogs(catalog_paths: list[str | Path]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for catalog_path in catalog_paths:
        documents.extend(build_source_documents_from_catalog(catalog_path))
    return drop_superseded_source_documents(dedupe_source_documents(documents))


def write_source_documents_jsonl(documents: list[SourceDocument], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(document.to_json())
            handle.write("\n")
