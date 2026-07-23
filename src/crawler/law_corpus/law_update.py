from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from crawler.law_corpus.acquire import acquire_sources
from crawler.law_corpus.catalog import load_sources
from crawler.law_corpus.corpus_store import (
    build_case_law_relations,
    build_law_relations,
    file_manifest,
    structure_law_documents,
    utc_now_iso,
    write_jsonl,
)
from crawler.law_corpus.models import AcquisitionSource
from crawler.law_corpus.parse_units import parse_source_documents, read_legal_units_jsonl
from crawler.law_corpus.source_documents import build_source_documents_from_catalogs


CORPUS_FILES = (
    "laws.jsonl",
    "legal_units.jsonl",
    "law_relations.jsonl",
    "gdprhub_cases.jsonl",
    "case_law_relations.jsonl",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _duplicate_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def load_unique_sources(catalog_paths: list[str | Path]) -> list[AcquisitionSource]:
    sources: list[AcquisitionSource] = []
    for catalog_path in catalog_paths:
        sources.extend(load_sources(catalog_path))
    if not sources:
        raise ValueError("The supplied catalogs contain no [[sources]] entries.")
    duplicates = _duplicate_values(source.doc_id for source in sources)
    if duplicates:
        raise ValueError("Duplicate doc_id values in catalogs: " + ", ".join(duplicates))
    return sources


def _validate_base_corpus(corpus_dir: Path) -> None:
    missing = [name for name in CORPUS_FILES if not (corpus_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Base corpus is incomplete at {corpus_dir}; missing: {', '.join(missing)}"
        )


def _validate_output_target(output_dir: Path, base_dir: Path) -> None:
    if output_dir.resolve() == base_dir.resolve():
        raise ValueError(
            "Output corpus must differ from the base corpus; in-place updates are disabled."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )


def add_laws(
    *,
    catalog_paths: list[str | Path],
    base_corpus_dir: str | Path = "corpus/structured",
    output_corpus_dir: str | Path = "corpus/structured_candidate",
    acquire: bool = True,
    manual_report_path: str | Path = "corpus/raw/manual_fetch.new_laws.md",
    timeout: int = 60,
) -> dict[str, Any]:
    """Acquire, parse, and merge new laws into a staged corpus directory.

    Existing law identifiers cannot be replaced. The base corpus is never modified.
    """

    catalogs = [Path(path) for path in catalog_paths]
    sources = load_unique_sources(catalogs)
    base_dir = Path(base_corpus_dir)
    output_dir = Path(output_corpus_dir)
    _validate_base_corpus(base_dir)
    _validate_output_target(output_dir, base_dir)

    existing_laws = _read_jsonl(base_dir / "laws.jsonl")
    existing_doc_ids = {str(record["doc_id"]) for record in existing_laws}
    incoming_doc_ids = {source.doc_id for source in sources}
    conflicts = sorted(existing_doc_ids & incoming_doc_ids)
    if conflicts:
        raise ValueError(
            "Catalog doc_id already exists in the base corpus: "
            + ", ".join(conflicts)
            + ". Use a versioned doc_id for a new legal version."
        )

    acquisition_results = []
    if acquire:
        acquisition_results = acquire_sources(
            sources,
            manual_report_path=manual_report_path,
            timeout=timeout,
        )
    unavailable = [source.doc_id for source in sources if not Path(source.target_path).is_file()]
    if unavailable:
        raise FileNotFoundError(
            "Raw files are unavailable after acquisition: "
            + ", ".join(sorted(unavailable))
            + f". See {manual_report_path} for manual-fetch instructions."
        )

    documents = build_source_documents_from_catalogs(catalogs)
    built_doc_ids = {document.doc_id for document in documents}
    omitted = sorted(incoming_doc_ids - built_doc_ids)
    if omitted:
        raise ValueError("No SourceDocument was built for: " + ", ".join(omitted))
    new_units = parse_source_documents(documents, require_all=True)

    existing_units = read_legal_units_jsonl(base_dir / "legal_units.jsonl")
    merged_units = sorted(
        [*existing_units, *new_units],
        key=lambda unit: (unit.source_doc_id, unit.unit_id),
    )
    duplicate_unit_ids = _duplicate_values(unit.unit_id for unit in merged_units)
    if duplicate_unit_ids:
        raise ValueError("Duplicate unit_id values after merge: " + ", ".join(duplicate_unit_ids))

    new_laws = structure_law_documents(documents, new_units)
    merged_laws = sorted([*existing_laws, *new_laws], key=lambda record: str(record["doc_id"]))
    cases = _read_jsonl(base_dir / "gdprhub_cases.jsonl")
    law_relations = build_law_relations(merged_units)
    case_law_relations = build_case_law_relations(cases, merged_units)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "laws.jsonl": write_jsonl(merged_laws, output_dir / "laws.jsonl"),
        "legal_units.jsonl": write_jsonl(merged_units, output_dir / "legal_units.jsonl"),
        "law_relations.jsonl": write_jsonl(law_relations, output_dir / "law_relations.jsonl"),
        "gdprhub_cases.jsonl": write_jsonl(cases, output_dir / "gdprhub_cases.jsonl"),
        "case_law_relations.jsonl": write_jsonl(
            case_law_relations,
            output_dir / "case_law_relations.jsonl",
        ),
    }
    report = {
        "generated_at": utc_now_iso(),
        "operation": "add_laws",
        "base_corpus": str(base_dir),
        "output_corpus": str(output_dir),
        "catalogs": [str(path) for path in catalogs],
        "new_doc_ids": sorted(incoming_doc_ids),
        "new_legal_unit_count": len(new_units),
        "acquisition": [asdict(result) for result in acquisition_results],
        "record_counts": counts,
    }
    report_path = output_dir / "update_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest_files = [
        file_manifest(output_dir / name, record_count=counts[name]) for name in CORPUS_FILES
    ]
    manifest_files.append(file_manifest(report_path))
    manifest = {
        "schema_version": "2.0.0",
        "generated_at": utc_now_iso(),
        "files": manifest_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relation_targets(records: list[Mapping[str, Any]], key: str) -> set[str]:
    return {str(record[key]) for record in records if record.get(key)}


def validate_corpus(corpus_dir: str | Path) -> dict[str, Any]:
    corpus = Path(corpus_dir)
    errors: list[str] = []
    missing = [name for name in (*CORPUS_FILES, "manifest.json") if not (corpus / name).is_file()]
    if missing:
        return {
            "valid": False,
            "corpus_dir": str(corpus),
            "errors": ["Missing files: " + ", ".join(missing)],
        }

    laws = _read_jsonl(corpus / "laws.jsonl")
    units = read_legal_units_jsonl(corpus / "legal_units.jsonl")
    law_relations = _read_jsonl(corpus / "law_relations.jsonl")
    cases = _read_jsonl(corpus / "gdprhub_cases.jsonl")
    case_law_relations = _read_jsonl(corpus / "case_law_relations.jsonl")

    doc_ids = [str(record.get("doc_id") or "") for record in laws]
    unit_ids = [unit.unit_id for unit in units]
    case_ids = [str(record.get("case_id") or "") for record in cases]
    for label, values in (("doc_id", doc_ids), ("unit_id", unit_ids), ("case_id", case_ids)):
        duplicates = _duplicate_values(values)
        if duplicates:
            errors.append(f"Duplicate {label}: {', '.join(duplicates[:20])}")
        if "" in values:
            errors.append(f"Empty {label} value")

    doc_id_set = set(doc_ids)
    unit_id_set = set(unit_ids)
    case_id_set = set(case_ids)
    unknown_unit_docs = sorted({unit.source_doc_id for unit in units} - doc_id_set)
    if unknown_unit_docs:
        errors.append("Legal units reference unknown laws: " + ", ".join(unknown_unit_docs))

    unknown_law_sources = sorted(_relation_targets(law_relations, "source_unit_id") - unit_id_set)
    unknown_law_targets = sorted(_relation_targets(law_relations, "target_unit_id") - unit_id_set)
    unknown_case_sources = sorted(_relation_targets(case_law_relations, "case_id") - case_id_set)
    unknown_case_targets = sorted(
        _relation_targets(case_law_relations, "target_unit_id") - unit_id_set
    )
    if unknown_law_sources:
        errors.append(
            "Law relations have unknown source units: " + ", ".join(unknown_law_sources[:20])
        )
    if unknown_law_targets:
        errors.append(
            "Law relations have unknown target units: " + ", ".join(unknown_law_targets[:20])
        )
    if unknown_case_sources:
        errors.append("Case relations have unknown cases: " + ", ".join(unknown_case_sources[:20]))
    if unknown_case_targets:
        errors.append(
            "Case relations have unknown target units: " + ", ".join(unknown_case_targets[:20])
        )

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    manifest_entries = {entry["path"]: entry for entry in manifest.get("files", [])}
    expected_counts = {
        "laws.jsonl": len(laws),
        "legal_units.jsonl": len(units),
        "law_relations.jsonl": len(law_relations),
        "gdprhub_cases.jsonl": len(cases),
        "case_law_relations.jsonl": len(case_law_relations),
    }
    for required_filename in expected_counts:
        if required_filename not in manifest_entries:
            errors.append(f"Manifest entry missing: {required_filename}")

    for filename, entry in manifest_entries.items():
        relative_path = Path(str(filename))
        if relative_path.is_absolute() or len(relative_path.parts) != 1:
            errors.append(f"Unsafe manifest path: {filename}")
            continue
        manifest_file = corpus / relative_path
        if not manifest_file.is_file():
            errors.append(f"Manifest file missing: {filename}")
            continue
        expected_count = expected_counts.get(filename)
        if expected_count is not None and entry.get("records") != expected_count:
            errors.append(
                f"Manifest record count mismatch for {filename}: "
                f"expected {expected_count}, found {entry.get('records')}"
            )
        if entry.get("bytes") != manifest_file.stat().st_size:
            errors.append(f"Manifest byte count mismatch for {filename}")
        actual_sha256 = _sha256_file(manifest_file)
        if entry.get("sha256") != actual_sha256:
            errors.append(f"Manifest SHA-256 mismatch for {filename}")

    return {
        "valid": not errors,
        "corpus_dir": str(corpus),
        "counts": expected_counts,
        "errors": errors,
    }


def describe_catalogs(catalog_paths: list[str | Path]) -> dict[str, Any]:
    sources = load_unique_sources(catalog_paths)
    return {
        "source_count": len(sources),
        "sources": [
            {
                "doc_id": source.doc_id,
                "title": source.title,
                "jurisdiction": source.jurisdiction,
                "law_family": source.law_family,
                "download_mode": source.download_mode,
                "target_path": source.target_path,
            }
            for source in sources
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="law-corpus-tool",
        description="Acquire and stage new laws for the legal retrieval corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="Inspect one or more law catalogs.")
    catalog_parser.add_argument("--catalog", action="append", required=True)

    add_parser = subparsers.add_parser("add", help="Acquire, parse, and merge new laws.")
    add_parser.add_argument("--catalog", action="append", required=True)
    add_parser.add_argument("--base-corpus", default="corpus/structured")
    add_parser.add_argument("--output-corpus", default="corpus/structured_candidate")
    add_parser.add_argument("--manual-report", default="corpus/raw/manual_fetch.new_laws.md")
    add_parser.add_argument("--timeout", type=int, default=60)
    add_parser.add_argument(
        "--skip-acquire",
        action="store_true",
        help="Use raw files already present at each catalog target_path.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a staged corpus.")
    validate_parser.add_argument("--corpus-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "catalog":
        result = describe_catalogs(args.catalog)
    elif args.command == "add":
        result = add_laws(
            catalog_paths=args.catalog,
            base_corpus_dir=args.base_corpus,
            output_corpus_dir=args.output_corpus,
            acquire=not args.skip_acquire,
            manual_report_path=args.manual_report,
            timeout=args.timeout,
        )
    else:
        result = validate_corpus(args.corpus_dir)
        if not result["valid"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
