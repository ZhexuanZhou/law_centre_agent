from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crawler.law_corpus.acquire import acquire_sources  # noqa: E402
from crawler.law_corpus.catalog import load_sources  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="corpus/sources/laws.seed.toml")
    parser.add_argument("--manual-report", default="corpus/raw/manual_fetch.md")
    args = parser.parse_args()

    sources = load_sources(Path(args.catalog))
    results = acquire_sources(sources, manual_report_path=args.manual_report)
    for result in results:
        print(f"{result.doc_id}\t{result.status}\t{result.target_path}")


if __name__ == "__main__":
    main()
