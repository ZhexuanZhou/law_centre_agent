from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from crawler.law_corpus.extract_text import extract_text_from_file  # noqa: E402


DEFAULT_LAWS = "corpus/structured/laws.jsonl"
DEFAULT_OUT = "corpus/structured/source_remote_check.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laws", default=DEFAULT_LAWS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    laws = [json.loads(line) for line in Path(args.laws).read_text(encoding="utf-8").splitlines()]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_check_source, law, timeout=args.timeout): law["doc_id"] for law in laws
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['doc_id']}: {result['status']} "
                f"http={result.get('http_status')} exact_bytes={result.get('exact_byte_match')}"
            )

    results.sort(key=lambda item: item["doc_id"])
    summary = {
        "checked": len(results),
        "reachable": sum(item["status"] == "reachable" for item in results),
        "indeterminate": sum(item["status"] == "indeterminate" for item in results),
        "exact_byte_matches": sum(item.get("exact_byte_match") is True for item in results),
        "different_remote_bytes": sum(item.get("exact_byte_match") is False for item in results),
        "exact_extracted_text_matches": sum(
            item.get("exact_extracted_text_match") is True for item in results
        ),
        "failed": sum(item["status"] == "failed" for item in results),
    }
    report = {
        "checked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "method": "HTTP GET of catalog source_url; SHA-256 comparison with the local raw file",
        "summary": summary,
        "results": results,
        "interpretation": (
            "An exact byte mismatch can be caused by a dynamic HTML wrapper or a newer current "
            "version; it requires text-level review and is not by itself proof of incorrect law text."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _check_source(law: dict, *, timeout: int) -> dict:
    target_path = Path(law["source_metadata"]["target_path"])
    local_sha256 = _sha256_file(target_path)
    result = {
        "doc_id": law["doc_id"],
        "source_url": law["source_url"],
        "local_path": str(target_path),
        "local_bytes": target_path.stat().st_size,
        "local_sha256": local_sha256,
    }
    try:
        response = requests.get(
            law["source_url"],
            headers={"User-Agent": "law-centre-corpus-validator/0.1 research"},
            timeout=timeout,
        )
        response.raise_for_status()
        remote_bytes = response.content
        if not remote_bytes:
            result.update(
                {
                    "status": "indeterminate",
                    "http_status": response.status_code,
                    "final_url": response.url,
                    "content_type": response.headers.get("Content-Type"),
                    "remote_bytes": 0,
                    "exact_byte_match": None,
                    "exact_extracted_text_match": None,
                    "reason": "successful HTTP status returned an empty response body",
                }
            )
            return result
        remote_sha256 = hashlib.sha256(remote_bytes).hexdigest()
        remote_text = _extract_remote_text(remote_bytes, suffix=target_path.suffix)
        local_text = str(law.get("raw_text") or "")
        result.update(
            {
                "status": "reachable",
                "http_status": response.status_code,
                "final_url": response.url,
                "content_type": response.headers.get("Content-Type"),
                "remote_bytes": len(remote_bytes),
                "remote_sha256": remote_sha256,
                "exact_byte_match": remote_sha256 == local_sha256,
                "remote_extracted_text_chars": len(remote_text),
                "remote_extracted_text_sha256": hashlib.sha256(
                    remote_text.encode("utf-8")
                ).hexdigest(),
                "exact_extracted_text_match": remote_text == local_text,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "exact_byte_match": None,
            }
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_remote_text(content: bytes, *, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(content)
        handle.flush()
        return extract_text_from_file(handle.name)


if __name__ == "__main__":
    main()
