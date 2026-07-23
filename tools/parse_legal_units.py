from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crawler.law_corpus.parse_units import (  # noqa: E402
    parse_source_documents,
    read_source_documents_jsonl_many,
    write_legal_units_jsonl,
)


DEFAULT_SOURCE_DOCUMENTS = "corpus/normalized/source_documents.full.jsonl"
DEFAULT_OUT = "corpus/parsed/legal_units.full.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-documents",
        action="append",
        default=None,
        help="SourceDocument JSONL path. Repeat to parse a merged corpus in order.",
    )
    parser.add_argument("--out")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any input SourceDocument produces zero LegalUnit records.",
    )
    args = parser.parse_args()

    source_document_paths = args.source_documents or [DEFAULT_SOURCE_DOCUMENTS]
    if args.out is None and _requires_explicit_out(source_document_paths):
        parser.error("--out is required when parsing non-canonical or multi-input documents")
    out = args.out or DEFAULT_OUT
    documents = read_source_documents_jsonl_many(source_document_paths)
    units = parse_source_documents(documents, require_all=args.require_all)
    write_legal_units_jsonl(units, out)
    print(f"wrote {len(units)} legal units to {out}")


def _requires_explicit_out(source_document_paths: list[str | Path]) -> bool:
    if len(source_document_paths) != 1:
        return True
    return _repo_relative_path(source_document_paths[0]) != _repo_relative_path(
        DEFAULT_SOURCE_DOCUMENTS
    )


def _repo_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        return candidate.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False))
    except ValueError:
        return candidate.resolve(strict=False)


if __name__ == "__main__":
    main()
