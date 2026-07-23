from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
import re
import time
from typing import Protocol
from urllib.parse import quote

from bs4 import BeautifulSoup, Tag
import requests

from crawler.law_corpus.case_models import CaseDocument, CaseSegment, CaseSegmentType
from crawler.law_corpus.parsers.base import stable_id


GDPRHUB_API_URL = "https://gdprhub.eu/api.php"
GDPRHUB_PAGE_URL = "https://gdprhub.eu/index.php?title={title}"
COUNTRY_NAME_TO_JURISDICTION = {
    "austria": "AT",
    "belgium": "BE",
    "bulgaria": "BG",
    "croatia": "HR",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "estonia": "EE",
    "european union": "EU",
    "eu": "EU",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "greece": "GR",
    "hungary": "HU",
    "iceland": "IS",
    "ireland": "IE",
    "italy": "IT",
    "latvia": "LV",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "luxembourg": "LU",
    "malta": "MT",
    "netherlands": "NL",
    "the netherlands": "NL",
    "norway": "NO",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "slovakia": "SK",
    "slovenia": "SI",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "uk": "UK",
    "united kingdom": "UK",
}
AUTHORITY_PREFIX_TO_JURISDICTION = {
    "bvwg": "AT",
    "ce": "FR",
    "cjeu": "EU",
    "edpb": "EU",
}
NON_CASE_TITLES = frozenset(
    {
        "Accurate titles for the decisions of DPAs",
        "About GDPRhub",
        "Advanced Search",
    }
)


class HttpGetSession(Protocol):
    def get(self, url: str, params: dict, timeout: int, headers: dict):
        raise NotImplementedError


class GDPRhubClient:
    def __init__(
        self,
        *,
        api_url: str = GDPRHUB_API_URL,
        session: HttpGetSession | None = None,
        timeout: int = 60,
        user_agent: str = "law-centre-crawler/0.1 research",
        max_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self.api_url = api_url
        self.session = session or requests
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def fetch_case_page(self, title: str) -> CaseDocument:
        payload = self._get(
            {
                "action": "parse",
                "page": title,
                "prop": "text|categories|externallinks",
                "format": "json",
                "formatversion": "2",
                "redirects": "1",
            }
        )
        parsed = payload["parse"]
        pageid = parsed.get("pageid")
        canonical_title = parsed.get("title", title)
        html_value = parsed.get("text", "")
        raw_html = html_value.get("*", "") if isinstance(html_value, dict) else str(html_value)
        categories = [
            _category_name(item) for item in parsed.get("categories", []) if _category_name(item)
        ]
        external_links = [str(link) for link in parsed.get("externallinks", [])]
        metadata = extract_gdprhub_case_metadata(raw_html)
        jurisdiction = infer_gdprhub_jurisdiction(canonical_title, categories, metadata)
        return CaseDocument(
            case_id=f"gdprhub:{pageid or stable_id(canonical_title)}",
            source_type="case_summary",
            title=canonical_title,
            source_url=GDPRHUB_PAGE_URL.format(title=quote(canonical_title.replace(" ", "_"))),
            language="en",
            raw_html=raw_html,
            raw_text=extract_gdprhub_case_text(raw_html),
            categories=categories,
            external_links=external_links,
            retrieved_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            jurisdiction=jurisdiction,
            metadata=metadata,
        )

    def list_category_members(self, category: str, *, limit: int = 50) -> list[str]:
        titles: list[str] = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": str(min(limit, 500)),
            "format": "json",
            "formatversion": "2",
        }
        while len(titles) < limit:
            payload = self._get(params)
            titles.extend(
                item["title"]
                for item in payload.get("query", {}).get("categorymembers", [])
                if item.get("ns") == 0
                and item.get("title")
                and is_gdprhub_case_title(item["title"])
            )
            continuation = payload.get("continue", {}).get("cmcontinue")
            if not continuation:
                break
            params["cmcontinue"] = continuation
        return titles[:limit]

    def list_all_case_titles(self, *, limit: int | None = None) -> list[str]:
        titles: list[str] = []
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": "0",
            "aplimit": "500",
            "format": "json",
            "formatversion": "2",
        }
        while limit is None or len(titles) < limit:
            payload = self._get(params)
            for item in payload.get("query", {}).get("allpages", []):
                title = item.get("title")
                if item.get("ns") == 0 and title and is_gdprhub_case_title(title):
                    titles.append(title)
                    if limit is not None and len(titles) >= limit:
                        break
            if limit is not None and len(titles) >= limit:
                break
            continuation = payload.get("continue", {}).get("apcontinue")
            if not continuation:
                break
            params["apcontinue"] = continuation
        return titles if limit is None else titles[:limit]

    def _get(self, params: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    self.api_url,
                    params=params,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                if self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        raise last_error or RuntimeError("GDPRhub request failed")


def is_gdprhub_case_title(title: str) -> bool:
    normalized = title.strip()
    if not normalized or normalized in NON_CASE_TITLES:
        return False
    excluded_prefixes = (
        "-",
        "Article ",
        "Category:",
        "File:",
        "GDPRhub",
        "Template:",
        "User:",
        "Welcome",
    )
    if normalized.startswith(excluded_prefixes):
        return False
    return " - " in normalized


def _looks_like_case_title(title: str) -> bool:
    return is_gdprhub_case_title(title)


def dedupe_case_documents(documents: list[CaseDocument]) -> list[CaseDocument]:
    seen: dict[str, CaseDocument] = {}
    deduped: list[CaseDocument] = []
    for document in documents:
        existing = seen.get(document.case_id)
        if existing is None:
            seen[document.case_id] = document
            deduped.append(document)
            continue
        if _case_document_content_key(existing) != _case_document_content_key(document):
            raise ValueError(
                f"Conflicting duplicate GDPRhub case document case_id={document.case_id}"
            )
    return deduped


def _case_document_content_key(document: CaseDocument) -> tuple[object, ...]:
    return (
        document.case_id,
        document.source_type,
        document.title,
        document.source_url,
        document.language,
        document.raw_html,
        document.categories,
        document.external_links,
        document.raw_text,
        document.metadata,
    )


def infer_gdprhub_jurisdiction(
    title: str,
    categories: Iterable[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> str | None:
    metadata = metadata or {}

    for key in ("jurisdiction", "country"):
        jurisdiction = _jurisdiction_from_text(metadata.get(key))
        if jurisdiction:
            return jurisdiction

    authority = _case_authority_prefix(title)
    metadata_authority = str(metadata.get("authority") or "").strip()
    for candidate_authority in (authority, metadata_authority):
        jurisdiction = _jurisdiction_from_authority(candidate_authority)
        if jurisdiction:
            return jurisdiction

    return _jurisdiction_from_categories(categories)


def _case_authority_prefix(title: str) -> str:
    return title.split(" - ", 1)[0].strip()


def _jurisdiction_from_authority(authority: str) -> str | None:
    if not authority:
        return None
    for country_label in re.findall(r"\(([^()]*)\)", authority):
        jurisdiction = _jurisdiction_from_text(country_label)
        if jurisdiction:
            return jurisdiction
    return AUTHORITY_PREFIX_TO_JURISDICTION.get(_normalized_label(authority))


def _jurisdiction_from_categories(categories: Iterable[str]) -> str | None:
    jurisdictions = {
        jurisdiction
        for category in categories
        if (jurisdiction := _jurisdiction_from_text(category))
    }
    national_jurisdictions = jurisdictions - {"EU"}
    if len(national_jurisdictions) == 1:
        return next(iter(national_jurisdictions))
    if len(jurisdictions) == 1:
        return next(iter(jurisdictions))
    return None


def _jurisdiction_from_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z0-9]{1,8})?", text):
        return text.upper()
    return COUNTRY_NAME_TO_JURISDICTION.get(_normalized_label(text))


def _normalized_label(value: str) -> str:
    value = re.sub(r"^category:", "", value, flags=re.IGNORECASE)
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def parse_gdprhub_case_segments(document: CaseDocument) -> list[CaseSegment]:
    soup = BeautifulSoup(document.raw_html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    parts: list[tuple[str, CaseSegmentType, str]] = []
    active_heading: str | None = None
    active_type: CaseSegmentType = "unknown"
    active_text: list[str] = []

    for child in root.children:
        if not isinstance(child, Tag):
            continue
        if child.name in {"h2", "h3"}:
            if active_heading and active_text:
                parts.append((active_heading, active_type, _clean_text(" ".join(active_text))))
            active_heading = _heading_text(child)
            active_type = _segment_type_for_heading(active_heading)
            active_text = []
            continue
        if active_heading:
            text = _clean_text(child.get_text(" ", strip=True))
            if text:
                active_text.append(text)

    if active_heading and active_text:
        parts.append((active_heading, active_type, _clean_text(" ".join(active_text))))

    document_text = "\n\n".join(f"{heading}\n{text}" for heading, _, text in parts)
    segments: list[CaseSegment] = []
    search_from = 0
    for heading, segment_type, text in parts:
        char_start = document_text.find(text, search_from)
        if char_start < 0:
            char_start = 0
        char_end = char_start + len(text)
        search_from = char_end
        segments.append(
            CaseSegment(
                segment_id=f"{document.case_id}:segment:{stable_id(heading, text[:80])}",
                source_case_id=document.case_id,
                source_url=document.source_url,
                segment_type=segment_type,
                heading=heading,
                text=text,
                char_start=char_start,
                char_end=char_end,
                language=document.language,
            )
        )
    return segments


def extract_gdprhub_case_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for tag in root.select("script, style, noscript"):
        tag.decompose()
    return _clean_text(root.get_text(" ", strip=True))


def extract_gdprhub_case_metadata(raw_html: str) -> dict[str, object]:
    soup = BeautifulSoup(raw_html, "html.parser")
    table = soup.select_one("table.wikitable")
    if table is None:
        return {}

    metadata: dict[str, object] = {}
    label_map = {
        "authority": "authority",
        "jurisdiction": "jurisdiction",
        "relevant law": "relevant_laws",
        "type": "case_type",
        "outcome": "outcome",
        "started": "started",
        "decided": "decided",
        "published": "published",
        "fine": "fine",
        "parties": "parties",
        "national case number/name": "national_case_number",
        "european case law identifier": "ecli",
        "appeal": "appeal",
        "original language(s)": "original_languages",
        "original source": "original_source",
    }
    for row in table.select("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = _clean_label(cells[0].get_text(" ", strip=True))
        key = label_map.get(label)
        if not key:
            continue
        value_cell = cells[1]
        if key == "relevant_laws":
            metadata[key] = _cell_lines(value_cell)
        else:
            metadata[key] = _clean_text(value_cell.get_text(" ", strip=True))
        if key == "original_source":
            first_link = value_cell.find("a", href=True)
            if first_link:
                metadata["original_source_url"] = str(first_link["href"])
    industry_tags = _infer_industry_tags(raw_html, metadata)
    if industry_tags:
        metadata["industry_tags"] = industry_tags
    return {key: value for key, value in metadata.items() if value}


def _category_name(item) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    return str(item.get("*") or item.get("title") or item.get("category") or "")


def _heading_text(tag: Tag) -> str:
    for edit_section in tag.select(".mw-editsection"):
        edit_section.decompose()
    return _clean_text(tag.get_text(" ", strip=True))


def _clean_label(value: str) -> str:
    return _clean_text(value).rstrip(":").lower()


def _cell_lines(tag: Tag) -> list[str]:
    values: list[str] = []
    for child in tag.find_all(["a", "span"], recursive=True):
        text = _clean_text(child.get_text(" ", strip=True))
        if text and text not in values:
            values.append(text)
    if values:
        return values
    for br in tag.find_all("br"):
        br.replace_with("\n")
    return [
        _clean_text(part)
        for part in tag.get_text("\n", strip=True).splitlines()
        if _clean_text(part)
    ]


def _infer_industry_tags(raw_html: str, metadata: dict[str, object]) -> list[str]:
    soup = BeautifulSoup(raw_html, "html.parser")
    for selector in ["table", "#toc", "pre"]:
        for tag in soup.select(selector):
            tag.decompose()
    paragraph_text = " ".join(
        _clean_text(tag.get_text(" ", strip=True)) for tag in soup.find_all("p")
    )
    text = " ".join(
        [
            str(metadata.get("parties") or ""),
            str(metadata.get("case_type") or ""),
            paragraph_text[:5000],
        ]
    ).lower()
    rules = {
        "retail": [
            "retail",
            "department store",
            "supermarket",
            "shop",
            "store",
            "customer",
        ],
        "healthcare": ["hospital", "clinic", "patient", "medical", "healthcare"],
        "employment": ["employee", "employer", "workplace", "worker", "staff"],
        "finance": ["bank", "credit", "insurance", "payment", "financial"],
        "telecom": ["telecom", "telephone", "mobile operator", "sim card"],
        "education": ["school", "student", "university", "education"],
        "public_sector": ["municipality", "police", "government", "public authority"],
        "online_platform": ["platform", "app", "mobile app", "social network"],
    }
    return [
        tag
        for tag, keywords in rules.items()
        if any(_keyword_matches(text, keyword) for keyword in keywords)
    ]


def _keyword_matches(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _segment_type_for_heading(heading: str) -> CaseSegmentType:
    normalized = heading.strip().lower()
    if normalized in {"facts", "fact"}:
        return "background"
    if normalized in {"dispute", "issue", "issues"}:
        return "issue"
    if normalized in {"holding", "decision", "reasoning"}:
        return "reasoning"
    if normalized in {"comment", "comments"}:
        return "comment"
    if normalized in {"further resources", "resources"}:
        return "resources"
    return "unknown"
