from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crawler.law_corpus.case_models import CaseDocument, CaseSegment  # noqa: E402
from crawler.law_corpus.case_sources.gdprhub import (  # noqa: E402
    GDPRhubClient,
    dedupe_case_documents,
    parse_gdprhub_case_segments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", action="append", default=[])
    parser.add_argument("--titles-file", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--category-limit", type=int, default=20)
    parser.add_argument("--all-pages", action="store_true")
    parser.add_argument("--all-pages-limit", type=int)
    parser.add_argument("--from-case-documents", action="store_true")
    parser.add_argument(
        "--case-documents", default="corpus/normalized/gdprhub_case_documents.jsonl"
    )
    parser.add_argument("--existing-case-documents", action="append", default=[])
    parser.add_argument("--merged-case-documents")
    parser.add_argument("--case-segments", default="corpus/parsed/gdprhub_case_segments.jsonl")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--source-max-retries", type=int, default=5)
    parser.add_argument("--source-retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--source-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-segments", action="store_true")
    args = parser.parse_args()

    existing_documents = read_case_documents_jsonl_many(args.existing_case_documents)
    if args.from_case_documents:
        loaded_documents = read_case_documents_jsonl(args.case_documents)
        resumed_documents: list[CaseDocument] = []
        new_documents: list[CaseDocument] = []
        merged_documents = (
            dedupe_case_documents(existing_documents + loaded_documents)
            if args.merged_case_documents
            else [*existing_documents, *loaded_documents]
        )
        segment_documents = merged_documents if args.merged_case_documents else loaded_documents
        output_documents = loaded_documents
        titles: list[str] = []
    else:
        client = _make_client(args)
        titles = _discover_titles(
            client,
            explicit_titles=args.title,
            title_files=args.titles_file,
            categories=args.category,
            category_limit=args.category_limit,
            include_all_pages=args.all_pages,
            all_pages_limit=args.all_pages_limit,
        )
        loaded_documents = []
        resumed_documents = (
            dedupe_case_documents(read_case_documents_jsonl(args.case_documents))
            if args.resume
            else []
        )
        base_documents = dedupe_case_documents(existing_documents + resumed_documents)
        new_documents = []
        output_documents = resumed_documents
        merged_documents = dedupe_case_documents(existing_documents + output_documents)
        segment_documents = merged_documents if args.merged_case_documents else output_documents

    if args.dry_run:
        loaded_label = (
            f"loaded_documents={len(loaded_documents)}"
            if args.from_case_documents
            else f"resumed_documents={len(resumed_documents)}"
        )
        print(
            "dry_run "
            f"titles={len(titles)} "
            f"existing_documents={len(existing_documents)} "
            f"{loaded_label} "
            f"write_segments={not args.no_segments}"
        )
        return

    if not args.from_case_documents:
        documents_by_id = {document.case_id: document for document in base_documents}
        existing_titles = {document.title for document in base_documents}
        titles_to_fetch = [title for title in titles if title not in existing_titles]
        if not args.resume:
            _truncate_file(args.case_documents)
        for _, document, failure in _fetch_case_pages(
            _make_client(args),
            titles=titles_to_fetch,
            max_workers=args.source_workers,
        ):
            if failure is not None:
                print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
                continue
            existing_document = documents_by_id.get(document.case_id)
            if existing_document is not None:
                dedupe_case_documents([existing_document, document])
                continue
            documents_by_id[document.case_id] = document
            existing_titles.add(document.title)
            new_documents.append(document)
            _append_jsonl([document], args.case_documents)
        output_documents = dedupe_case_documents([*resumed_documents, *new_documents])
        merged_documents = dedupe_case_documents([*existing_documents, *output_documents])
        segment_documents = merged_documents if args.merged_case_documents else output_documents

    if args.from_case_documents and args.merged_case_documents:
        _write_jsonl(merged_documents, args.merged_case_documents)
    elif args.merged_case_documents:
        _write_jsonl(merged_documents, args.merged_case_documents)

    segment_count = 0
    if not args.no_segments:
        segments = _parse_segments(segment_documents)
        segment_count = len(segments)
        _write_jsonl(segments, args.case_segments)

    print(f"wrote case_documents={len(segment_documents)} case_segments={segment_count}")


def read_case_documents_jsonl(path: str | Path) -> list[CaseDocument]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    documents: list[CaseDocument] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(CaseDocument.from_json(line))
    return documents


def read_case_documents_jsonl_many(paths: list[str | Path]) -> list[CaseDocument]:
    documents: list[CaseDocument] = []
    for path in paths:
        documents.extend(read_case_documents_jsonl(path))
    return dedupe_case_documents(documents)


def _fetch_case_pages(
    client: GDPRhubClient,
    *,
    titles: list[str],
    max_workers: int,
):
    worker_count = max(1, max_workers)
    if worker_count == 1:
        for title in titles:
            yield _fetch_case_page(client, title)
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_fetch_case_page, client, title) for title in titles]
        for future in as_completed(futures):
            yield future.result()


def _fetch_case_page(client: GDPRhubClient, title: str):
    try:
        return title, client.fetch_case_page(title), None
    except Exception as exc:
        return (
            title,
            None,
            {
                "title": title,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )


def _make_client(args: argparse.Namespace) -> GDPRhubClient:
    return GDPRhubClient(
        timeout=args.request_timeout,
        max_retries=args.source_max_retries,
        retry_sleep_seconds=args.source_retry_sleep_seconds,
    )


def _discover_titles(
    client: GDPRhubClient,
    *,
    explicit_titles: list[str],
    title_files: list[str],
    categories: list[str],
    category_limit: int,
    include_all_pages: bool,
    all_pages_limit: int | None,
) -> list[str]:
    titles = list(explicit_titles)
    for title_file in title_files:
        titles.extend(_read_titles_file(title_file))
    if include_all_pages:
        titles.extend(client.list_all_case_titles(limit=all_pages_limit))
    for category in categories:
        titles.extend(client.list_category_members(category, limit=category_limit))
    deduped: list[str] = []
    seen: set[str] = set()
    for title in titles:
        if title not in seen:
            seen.add(title)
            deduped.append(title)
    return deduped


def _read_titles_file(path: str | Path) -> list[str]:
    titles: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("{"):
            payload = json.loads(value)
            value = str(payload.get("title", "")).strip()
        if value:
            titles.append(value)
    return titles


def _parse_segments(documents: list[CaseDocument]) -> list[CaseSegment]:
    segments: list[CaseSegment] = []
    for document in documents:
        segments.extend(parse_gdprhub_case_segments(document))
    return segments


def _write_jsonl(items, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.to_json())
            handle.write("\n")


def _append_jsonl(items, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.to_json())
            handle.write("\n")


def _truncate_file(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
