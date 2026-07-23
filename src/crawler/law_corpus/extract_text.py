from __future__ import annotations

from pathlib import Path
import re
from xml.etree import ElementTree

from bs4 import BeautifulSoup
from pypdf import PdfReader


UK_LEGISLATION_NAMESPACE = "http://www.legislation.gov.uk/namespaces/legislation"
UK_SECTION_URI_RE = re.compile(r"/section/(\d+[A-Z]?)$")
UK_SCHEDULE_PARAGRAPH_URI_RE = re.compile(r"/schedule/([^/]+)/paragraph/(\d+[A-Z]?)$")
UK_OMISSION_PLACEHOLDER_RE = re.compile(r"^(?:\.\s*)+$")
UK_PARTIAL_OMISSION_RE = re.compile(r"(?:\.\s*){8,}")


def extract_text_from_file(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return normalize_text(file_path.read_text(encoding="utf-8"))
    if suffix in {".html", ".htm"}:
        return extract_text_from_html(file_path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".xml":
        return extract_text_from_xml(file_path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    raise ValueError(f"Unsupported law source file type: {file_path}")


def extract_uk_legislation_metadata_from_file(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    if file_path.suffix.lower() != ".xml":
        return {}
    try:
        root = ElementTree.parse(file_path).getroot()
    except (ElementTree.ParseError, OSError):
        return {}
    return extract_uk_legislation_metadata(root)


def extract_uk_legislation_metadata(root: ElementTree.Element) -> dict[str, object]:
    if _namespace(root.tag) != UK_LEGISLATION_NAMESPACE or _local_name(root.tag) != "Legislation":
        return {}

    commentary_by_ref = {
        commentary_id: _element_text(element)
        for element in root.iter()
        if _local_name(element.tag) == "Commentary"
        if (commentary_id := element.attrib.get("id"))
    }
    sections: dict[str, dict[str, object]] = {}
    for group in root.iter():
        if _local_name(group.tag) != "P1group":
            continue
        p1 = _direct_child(group, "P1")
        if p1 is None:
            continue
        match = UK_SECTION_URI_RE.search(p1.attrib.get("DocumentURI", ""))
        if match is None:
            continue

        number = match.group(1)
        body = _uk_section_body_text(p1)
        commentary_refs = _uk_commentary_refs(group)
        commentaries = [
            {"ref": ref, "text": commentary_by_ref[ref]}
            for ref in commentary_refs
            if ref in commentary_by_ref
        ]
        commentary_texts = [str(item["text"]) for item in commentaries]
        is_omitted = bool(body) and UK_OMISSION_PLACEHOLDER_RE.fullmatch(body) is not None
        restriction_start_date = group.attrib.get("RestrictStartDate")
        effective_from = _uk_section_event_date(number, commentary_texts, "inserted")
        effective_to = None
        if is_omitted:
            effective_to = (
                _uk_section_event_date(number, commentary_texts, "omitted")
                or _uk_section_event_date(number, commentary_texts, "repealed")
                or restriction_start_date
            )
        sections[number] = {
            "source_uri": p1.attrib.get("DocumentURI"),
            "title": _element_text(_direct_child(group, "Title")),
            "status": "omitted" if is_omitted else "active",
            "effective_from": effective_from,
            "effective_to": effective_to,
            "restriction_start_date": restriction_start_date,
            "has_omitted_provisions": (
                not is_omitted and UK_PARTIAL_OMISSION_RE.search(body) is not None
            ),
            "commentaries": commentaries,
        }

    if not sections:
        return {}
    return {"schema_version": "1.0", "sections": sections}


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    australian_text = extract_text_from_australian_epub_html(soup)
    if australian_text:
        return normalize_text(australian_text)
    text = soup.get_text("\n")
    return normalize_text(text)


def extract_text_from_xml(xml_text: str) -> str:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return extract_text_from_html(xml_text)
    uk_text = extract_text_from_uk_legislation_xml(root)
    if uk_text:
        return normalize_text(uk_text)
    return normalize_text("\n".join(root.itertext()))


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return normalize_text("\n".join(pages))


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def extract_text_from_australian_epub_html(soup: BeautifulSoup) -> str | None:
    start = None
    for candidate in soup.find_all("p", class_="ActHead5"):
        if re.match(r"^\s*1\s+Short title\s*$", candidate.get_text(" ", strip=True)):
            start = candidate
            break
    if start is None:
        return None

    lines: list[str] = []
    for node in [start, *start.find_all_next("p")]:
        classes = set(node.get("class", []))
        if "ENotesHeading1" in classes:
            break
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        lines.extend(_australian_epub_lines(text, classes))

    if not lines:
        return None
    return "\n".join(lines)


def _australian_epub_lines(text: str, classes: set[str]) -> list[str]:
    compact = " ".join(text.split())
    if "ActHead1" in classes:
        schedule_match = re.match(r"^Schedule\s+(\d+)\s+([—-])\s+(.+)$", compact)
        if schedule_match:
            return [
                "Schedule",
                schedule_match.group(1),
                schedule_match.group(2),
                schedule_match.group(3),
            ]
        return [compact]
    if "ActHead5" in classes:
        app_match = re.match(
            r"^(\d{1,2})\s+Australian Privacy Principle\s+(.+)$",
            compact,
        )
        if app_match:
            return [app_match.group(1), "Australian Privacy Principle", app_match.group(2)]
        section_match = re.match(r"^(\d+[A-Z]{0,3})\s+(.+)$", compact)
        if section_match:
            return [section_match.group(1), section_match.group(2)]
    if "subsection" in classes:
        app_clause_match = re.match(r"^(\d{1,2}\.\d+)\s+(.+)$", compact)
        if app_clause_match:
            return [app_clause_match.group(1), app_clause_match.group(2)]
    return [compact]


def extract_text_from_uk_legislation_xml(root: ElementTree.Element) -> str | None:
    if _namespace(root.tag) != UK_LEGISLATION_NAMESPACE or _local_name(root.tag) != "Legislation":
        return None

    sections: list[str] = []
    for group in root.iter():
        if _local_name(group.tag) != "P1group":
            continue
        p1 = _direct_child(group, "P1")
        if p1 is None:
            continue
        match = UK_SECTION_URI_RE.search(p1.attrib.get("DocumentURI", ""))
        if match is None:
            continue
        title = _element_text(_direct_child(group, "Title"))
        number = _element_text(_direct_child(p1, "Pnumber")) or match.group(1)
        body = _uk_section_body_text(p1)
        parts = [number, title, body]
        section_text = "\n".join(part for part in parts if part)
        if section_text:
            sections.append(section_text)

    schedules = _extract_uk_schedule_texts(root)
    if not sections and not schedules:
        return None
    return "\n\n".join([*sections, *schedules])


def _extract_uk_schedule_texts(root: ElementTree.Element) -> list[str]:
    schedules: list[str] = []
    root_document_uri = root.attrib.get("DocumentURI", "").rstrip("/")
    for schedule in root.iter():
        if _local_name(schedule.tag) != "Schedule":
            continue
        schedule_uri = schedule.attrib.get("DocumentURI", "").rstrip("/")
        if not _is_primary_uk_schedule_uri(schedule_uri, root_document_uri):
            continue
        number = _normalize_uk_schedule_number(_element_text(_direct_child(schedule, "Number")))
        if not number:
            continue
        title = _element_text(_direct_child(schedule, "TitleBlock"))
        paragraph_texts = _uk_schedule_paragraph_texts(schedule)
        if paragraph_texts:
            parts = [number, title, *paragraph_texts]
        else:
            body = _element_text(_direct_child(schedule, "ScheduleBody"), wrap_pnumber=True)
            parts = [number, title, body]
        schedule_text = "\n".join(part for part in parts if part)
        if schedule_text:
            schedules.append(schedule_text)
    return schedules


def _uk_schedule_paragraph_texts(schedule: ElementTree.Element) -> list[str]:
    paragraphs: list[str] = []
    for p1 in schedule.iter():
        if _local_name(p1.tag) != "P1":
            continue
        match = UK_SCHEDULE_PARAGRAPH_URI_RE.search(p1.attrib.get("DocumentURI", ""))
        if match is None:
            continue
        number = _element_text(_direct_child(p1, "Pnumber")) or match.group(2)
        body = _uk_section_body_text(p1)
        paragraph = "\n".join(part for part in [f"Paragraph {number}", body] if part)
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def _normalize_uk_schedule_number(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"schedule\s+([A-Z]?\d+[A-Z]?)", value, flags=re.IGNORECASE)
    if match:
        return f"SCHEDULE {match.group(1).upper()}"
    return value.upper()


def _is_primary_uk_schedule_uri(schedule_uri: str, root_document_uri: str) -> bool:
    if not schedule_uri:
        return False
    if root_document_uri:
        return schedule_uri.startswith(f"{root_document_uri}/schedule/")
    return bool(re.search(r"/ukpga/\d+/\d+/schedule/[A-Z]?\d+[A-Z]?$", schedule_uri))


def _uk_section_body_text(p1: ElementTree.Element) -> str:
    body_parts = [
        _element_text(child, wrap_pnumber=True)
        for child in p1
        if _local_name(child.tag) != "Pnumber"
    ]
    return " ".join(part for part in body_parts if part)


def _uk_commentary_refs(element: ElementTree.Element) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for node in element.iter():
        values: list[str] = []
        if _local_name(node.tag) == "CommentaryRef":
            values.append(node.attrib.get("Ref", ""))
        values.append(node.attrib.get("CommentaryRef", ""))
        for value in values:
            for ref in value.split():
                if ref and ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    return refs


def _uk_section_event_date(
    section_number: str,
    commentaries: list[str],
    event: str,
) -> str | None:
    pattern = re.compile(
        rf"\bS\.\s*{re.escape(section_number)}\b[^.;()]{{0,80}}\b{re.escape(event)}\s*"
        r"\((\d{1,2})\.(\d{1,2})\.(\d{4})\)",
        flags=re.IGNORECASE,
    )
    for commentary in commentaries:
        match = pattern.search(commentary)
        if match:
            day, month, year = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def _direct_child(element: ElementTree.Element, local_name: str) -> ElementTree.Element | None:
    for child in element:
        if _local_name(child.tag) == local_name:
            return child
    return None


def _element_text(
    element: ElementTree.Element | None,
    *,
    wrap_pnumber: bool = False,
) -> str:
    if element is None:
        return ""
    parts: list[str] = []

    def walk(node: ElementTree.Element) -> None:
        local = _local_name(node.tag)
        if node.text and local != "CommentaryRef":
            text = node.text.strip()
            if text:
                parts.append(f"({text})" if wrap_pnumber and local == "Pnumber" else text)
        for child in node:
            if _local_name(child.tag) != "CommentaryRef":
                walk(child)
            if child.tail:
                tail = child.tail.strip()
                if tail:
                    parts.append(tail)

    walk(element)
    return " ".join(" ".join(parts).split())


def _namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
