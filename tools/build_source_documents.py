from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crawler.law_corpus.source_documents import (  # noqa: E402
    build_source_documents_from_catalogs,
    write_source_documents_jsonl,
)


DEFAULT_CATALOG = "corpus/sources/laws.seed.toml"
DEFAULT_OUT = "corpus/normalized/source_documents.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        action="append",
        default=None,
        help="Catalog TOML path. Repeat to build a merged corpus in order.",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    catalogs = args.catalog or [DEFAULT_CATALOG]
    if args.out is None and _requires_explicit_out(catalogs):
        parser.error("--out is required when building non-seed or multi-catalog outputs")
    out = args.out or DEFAULT_OUT
    documents = build_source_documents_from_catalogs(catalogs)
    write_source_documents_jsonl(documents, out)
    print(f"wrote {len(documents)} source documents to {out}")


def _requires_explicit_out(catalogs: list[str | Path]) -> bool:
    if len(catalogs) != 1:
        return True
    return _repo_relative_path(catalogs[0]) != _repo_relative_path(DEFAULT_CATALOG)


def _repo_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        return candidate.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False))
    except ValueError:
        return candidate.resolve(strict=False)


if __name__ == "__main__":
    main()
