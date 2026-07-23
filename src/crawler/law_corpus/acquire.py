from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
from typing import Protocol
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

from crawler.law_corpus.models import AcquisitionResult, AcquisitionSource


class HttpSession(Protocol):
    def get(self, url: str, timeout: int, headers: dict[str, str] | None = None): ...


BLOCKED_RESPONSE_MARKERS = (
    "Federal Register :: Request Access",
    "unblock.federalregister.gov",
)


def metadata_path_for(target_path: str | Path) -> Path:
    path = Path(target_path)
    return path.with_suffix(".metadata.json")


def write_metadata(source: AcquisitionSource) -> Path:
    metadata_path = metadata_path_for(source.target_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(source)
    payload["raw_path"] = source.target_path
    payload["source_url"] = source.url
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def acquire_sources(
    sources: list[AcquisitionSource],
    *,
    session: HttpSession | None = None,
    manual_report_path: str | Path = "outputs/acquisition/manual_fetch.md",
    timeout: int = 60,
) -> list[AcquisitionResult]:
    allow_curl_fallback = session is None
    http = session or requests.Session()
    results: list[AcquisitionResult] = []
    manual_items: list[AcquisitionSource] = []

    for source in sources:
        target = Path(source.target_path)
        metadata_path = write_metadata(source)

        if target.exists():
            results.append(
                AcquisitionResult(
                    doc_id=source.doc_id,
                    status="already_exists",
                    target_path=str(target),
                    metadata_path=str(metadata_path),
                    message="Raw file already exists.",
                )
            )
            continue

        if source.requires_manual_fetch:
            manual_items.append(source)
            results.append(
                AcquisitionResult(
                    doc_id=source.doc_id,
                    status="manual_required",
                    target_path=str(target),
                    metadata_path=str(metadata_path),
                    message="Manual fetch required by catalog.",
                )
            )
            continue

        try:
            response = http.get(source.url, timeout=timeout, headers=request_headers_for(source))
            response.raise_for_status()
            if not response.content:
                raise ValueError("Source returned an empty response body.")
            if is_blocked_access_response(response):
                raise ValueError("Source returned an access-gate page instead of legal text.")
            content = response.content
            if source.preferred_format == "html":
                content = expand_paginated_html(
                    content,
                    base_url=str(getattr(response, "url", source.url)),
                    session=http,
                    source=source,
                    timeout=timeout,
                )
                content = follow_embedded_document_iframe(
                    content,
                    base_url=str(getattr(response, "url", source.url)),
                    session=http,
                    source=source,
                    timeout=timeout,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            results.append(
                AcquisitionResult(
                    doc_id=source.doc_id,
                    status="downloaded",
                    target_path=str(target),
                    metadata_path=str(metadata_path),
                    message="Downloaded successfully.",
                )
            )
        except Exception as exc:
            if allow_curl_fallback:
                try:
                    content = download_with_curl(
                        source.url,
                        headers=request_headers_for(source),
                        timeout=timeout,
                    )
                    if not content:
                        raise ValueError("curl returned an empty response body.")
                    if is_blocked_access_content(source.url, content):
                        raise ValueError("curl returned an access-gate page instead of legal text.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    results.append(
                        AcquisitionResult(
                            doc_id=source.doc_id,
                            status="downloaded",
                            target_path=str(target),
                            metadata_path=str(metadata_path),
                            message="Downloaded successfully with curl fallback.",
                        )
                    )
                    continue
                except Exception as fallback_exc:
                    exc = RuntimeError(f"{exc}; curl fallback failed: {fallback_exc}")
            manual_items.append(source)
            results.append(
                AcquisitionResult(
                    doc_id=source.doc_id,
                    status="failed",
                    target_path=str(target),
                    metadata_path=str(metadata_path),
                    message=f"Download failed: {exc}",
                )
            )

    write_manual_report(manual_items, manual_report_path)
    return results


def is_blocked_access_response(response) -> bool:
    final_url = str(getattr(response, "url", ""))
    content = getattr(response, "content", b"")
    return is_blocked_access_content(final_url, content)


def is_blocked_access_content(final_url: str, content: bytes | str) -> bool:
    if isinstance(content, bytes):
        body = content[:20000].decode("utf-8", errors="ignore")
    else:
        body = str(content)[:20000]
    haystack = f"{final_url}\n{body}"
    return any(marker in haystack for marker in BLOCKED_RESPONSE_MARKERS)


def download_with_curl(url: str, *, headers: dict[str, str], timeout: int) -> bytes:
    curl_path = shutil.which("curl")
    if curl_path is None:
        raise RuntimeError("curl executable is not available")
    command = [
        curl_path,
        "--location",
        "--fail",
        "--show-error",
        "--silent",
        "--max-time",
        str(timeout),
    ]
    for name, value in headers.items():
        command.extend(["--header", f"{name}: {value}"])
    command.append(url)
    result = subprocess.run(command, check=True, capture_output=True)
    return result.stdout


def expand_paginated_html(
    content: bytes,
    *,
    base_url: str,
    session: HttpSession,
    source: AcquisitionSource,
    timeout: int,
) -> bytes:
    page_urls = pagination_urls(content, base_url)
    if not page_urls:
        return content

    fragments = [article_body_fragment(content)]
    for page_url in page_urls:
        response = session.get(page_url, timeout=timeout, headers=request_headers_for(source))
        response.raise_for_status()
        if not response.content:
            raise ValueError(f"Paginated source returned an empty response body: {page_url}")
        if is_blocked_access_response(response):
            raise ValueError(f"Paginated source returned an access-gate page: {page_url}")
        fragments.append(article_body_fragment(response.content))

    body = "\n".join(fragment for fragment in fragments if fragment.strip())
    return (
        f'<!doctype html><html><head><meta charset="utf-8"></head><body>\n{body}\n</body></html>\n'
    ).encode("utf-8")


def follow_embedded_document_iframe(
    content: bytes,
    *,
    base_url: str,
    session: HttpSession,
    source: AcquisitionSource,
    timeout: int,
) -> bytes:
    iframe_url = embedded_document_iframe_url(content, base_url)
    if iframe_url is None:
        return content
    response = session.get(iframe_url, timeout=timeout, headers=request_headers_for(source))
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"Embedded document iframe returned an empty response body: {iframe_url}")
    if is_blocked_access_response(response):
        raise ValueError(f"Embedded document iframe returned an access-gate page: {iframe_url}")
    return response.content


def embedded_document_iframe_url(content: bytes, base_url: str) -> str | None:
    soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
    iframe = soup.select_one("iframe#epubFrame[src]")
    if iframe is None:
        return None
    href = iframe.get("src")
    if not href:
        return None
    return urljoin(base_url, href)


def pagination_urls(content: bytes, base_url: str) -> list[str]:
    soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for link in soup.select("a.page-Article"):
        href = link.get("href")
        if not href:
            continue
        url = urljoin(base_url, href)
        if url == base_url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def article_body_fragment(content: bytes) -> str:
    soup = BeautifulSoup(content.decode("utf-8", errors="ignore"), "html.parser")
    for selector in [
        "#div_page_roll1",
        "#div_currpage",
        ".zwfenye",
        ".page-Article",
        ".nextpage",
    ]:
        for node in soup.select(selector):
            node.decompose()
    body = soup.select_one("#BodyLabel") or soup.select_one("#content") or soup.body or soup
    return str(body)


def request_headers_for(source: AcquisitionSource) -> dict[str, str]:
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
    if "sso.agc.gov.sg" not in source.url:
        user_agent = f"{user_agent} law-centre-crawler/0.1"
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if source.preferred_format == "xml":
        headers["Accept"] = "application/xml;notice=tree"
    elif source.preferred_format == "pdf":
        headers["Accept"] = "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8"
    return headers


def write_manual_report(sources: list[AcquisitionSource], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Manual Law Source Fetch List", ""]
    if not sources:
        lines.append("No manual fetch is required.")
    for source in sources:
        lines.extend(
            [
                f"## {source.doc_id}",
                "",
                f"- Title: {source.title}",
                f"- URL: {source.url}",
                f"- Save to: `{source.target_path}`",
                f"- Metadata sidecar: `{metadata_path_for(source.target_path)}`",
                f"- Instructions: {source.manual_instructions or 'Download the official text and save it to the target path.'}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
