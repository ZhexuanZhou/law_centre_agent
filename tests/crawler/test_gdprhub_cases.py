import json
import sys

from crawler.law_corpus.case_models import CaseDocument
from crawler.law_corpus.case_sources.gdprhub import (
    GDPRhubClient,
    dedupe_case_documents,
    infer_gdprhub_jurisdiction,
    is_gdprhub_case_title,
    parse_gdprhub_case_segments,
)

import tools.acquire_gdprhub_cases as acquire_gdprhub_cases


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.params: dict = {}

    def get(self, url: str, params: dict, timeout: int, headers: dict) -> FakeResponse:
        self.params = params
        return FakeResponse(self.payload)


class QueueSession:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.params_seen: list[dict] = []

    def get(self, url: str, params: dict, timeout: int, headers: dict) -> FakeResponse:
        self.params_seen.append(dict(params))
        return FakeResponse(self.payloads.pop(0))


class FlakySession(FakeSession):
    def __init__(self, payload: dict) -> None:
        super().__init__(payload)
        self.calls = 0

    def get(self, url: str, params: dict, timeout: int, headers: dict) -> FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient ssl eof")
        return super().get(url, params, timeout, headers)


def gdprhub_case_document(
    case_id: str,
    title: str,
    *,
    raw_html: str = "<div class='mw-parser-output'><h2>Facts</h2><p>Facts.</p></div>",
    raw_text: str = "Facts. ",
) -> CaseDocument:
    return CaseDocument(
        case_id=case_id,
        source_type="case_summary",
        title=title,
        source_url=f"https://gdprhub.eu/index.php?title={title}",
        language="en",
        raw_html=raw_html,
        raw_text=raw_text,
        categories=[],
        external_links=[],
        retrieved_at="2026-01-01T00:00:00Z",
        metadata={},
    )


def test_is_gdprhub_case_title_keeps_cases_and_excludes_navigation_titles():
    assert is_gdprhub_case_title("AEPD (Spain) - PS/00001/2020")
    assert is_gdprhub_case_title("CNIL (France) - SAN-2021-001")

    assert not is_gdprhub_case_title("Article 6 GDPR")
    assert not is_gdprhub_case_title("GDPRhub")
    assert not is_gdprhub_case_title("Template:Decision")
    assert not is_gdprhub_case_title("Welcome to GDPRhub")
    assert not is_gdprhub_case_title("Accurate titles for the decisions of DPAs")
    assert not is_gdprhub_case_title("About GDPRhub")
    assert not is_gdprhub_case_title("Advanced Search")


def test_infer_gdprhub_jurisdiction_from_title_country_and_authority_alias():
    assert infer_gdprhub_jurisdiction("AEPD (Spain) - PS/00001/2020") == "ES"
    assert infer_gdprhub_jurisdiction("ICO (UK) - Tuckers Solicitors LLP") == "UK"
    assert infer_gdprhub_jurisdiction("BVwG - W211 2222613-1") == "AT"
    assert infer_gdprhub_jurisdiction("EDPB - Binding Decision 1/2020") == "EU"


def test_infer_gdprhub_jurisdiction_uses_unique_country_category_as_fallback():
    assert (
        infer_gdprhub_jurisdiction(
            "Unknown Authority - Example",
            categories=["Article_5_GDPR", "European_Union", "France"],
        )
        == "FR"
    )
    assert (
        infer_gdprhub_jurisdiction(
            "Unknown Authority - Example",
            categories=["France", "Spain"],
        )
        is None
    )


def test_dedupe_case_documents_keeps_identical_duplicate_case_id():
    document = gdprhub_case_document("gdprhub:1", "AEPD (Spain) - TEST")

    assert dedupe_case_documents([document, document]) == [document]


def test_dedupe_case_documents_ignores_retrieval_timestamp_changes():
    first = gdprhub_case_document("gdprhub:1", "AEPD (Spain) - TEST")
    second = CaseDocument(**{**first.__dict__, "retrieved_at": "2026-01-02T00:00:00Z"})

    assert dedupe_case_documents([first, second]) == [first]


def test_dedupe_case_documents_ignores_jurisdiction_enrichment_changes():
    first = gdprhub_case_document("gdprhub:1", "AEPD (Spain) - TEST")
    second = CaseDocument(**{**first.__dict__, "jurisdiction": "ES"})

    assert dedupe_case_documents([first, second]) == [first]


def test_dedupe_case_documents_rejects_conflicting_duplicate_case_id():
    first = gdprhub_case_document("gdprhub:1", "AEPD (Spain) - TEST")
    second = CaseDocument(**{**first.__dict__, "raw_text": "Different facts."})

    try:
        dedupe_case_documents([first, second])
    except ValueError as exc:
        assert "gdprhub:1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_acquire_gdprhub_cases_writes_incremental_and_merged_outputs(
    tmp_path,
    monkeypatch,
):
    existing = gdprhub_case_document("gdprhub:existing", "AEPD (Spain) - EXISTING")
    new = gdprhub_case_document("gdprhub:new", "CNIL (France) - NEW")
    existing_path = tmp_path / "existing.jsonl"
    incremental_path = tmp_path / "incremental.jsonl"
    merged_path = tmp_path / "merged.jsonl"
    case_segments_path = tmp_path / "case_segments.jsonl"
    incremental_path.write_text("stale output\n", encoding="utf-8")
    acquire_gdprhub_cases._write_jsonl([existing], existing_path)

    class FakeClient:
        def __init__(self) -> None:
            self.fetched_titles: list[str] = []

        def fetch_case_page(self, title: str) -> CaseDocument:
            self.fetched_titles.append(title)
            if title == new.title:
                return new
            raise AssertionError(f"Unexpected fetch: {title}")

    fake_client = FakeClient()
    monkeypatch.setattr(acquire_gdprhub_cases, "_make_client", lambda args: fake_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_gdprhub_cases.py",
            "--title",
            existing.title,
            "--title",
            new.title,
            "--existing-case-documents",
            str(existing_path),
            "--case-documents",
            str(incremental_path),
            "--merged-case-documents",
            str(merged_path),
            "--case-segments",
            str(case_segments_path),
        ],
    )

    acquire_gdprhub_cases.main()

    assert fake_client.fetched_titles == [new.title]
    assert acquire_gdprhub_cases.read_case_documents_jsonl(incremental_path) == [new]
    assert acquire_gdprhub_cases.read_case_documents_jsonl(merged_path) == [existing, new]
    segment_case_ids = [
        json.loads(line)["source_case_id"]
        for line in case_segments_path.read_text(encoding="utf-8").splitlines()
    ]
    assert segment_case_ids == [existing.case_id, new.case_id]


def test_acquire_gdprhub_cases_segments_follow_incremental_output_without_merged(
    tmp_path,
    monkeypatch,
):
    existing = gdprhub_case_document("gdprhub:existing", "AEPD (Spain) - EXISTING")
    new = gdprhub_case_document("gdprhub:new", "CNIL (France) - NEW")
    existing_path = tmp_path / "existing.jsonl"
    incremental_path = tmp_path / "incremental.jsonl"
    case_segments_path = tmp_path / "case_segments.jsonl"
    acquire_gdprhub_cases._write_jsonl([existing], existing_path)

    class FakeClient:
        def fetch_case_page(self, title: str) -> CaseDocument:
            if title == new.title:
                return new
            raise AssertionError(f"Unexpected fetch: {title}")

    monkeypatch.setattr(acquire_gdprhub_cases, "_make_client", lambda args: FakeClient())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_gdprhub_cases.py",
            "--title",
            existing.title,
            "--title",
            new.title,
            "--existing-case-documents",
            str(existing_path),
            "--case-documents",
            str(incremental_path),
            "--case-segments",
            str(case_segments_path),
        ],
    )

    acquire_gdprhub_cases.main()

    assert acquire_gdprhub_cases.read_case_documents_jsonl(incremental_path) == [new]
    segment_case_ids = [
        json.loads(line)["source_case_id"]
        for line in case_segments_path.read_text(encoding="utf-8").splitlines()
    ]
    assert segment_case_ids == [new.case_id]


def test_from_case_documents_preserves_segment_only_duplicate_compatibility(
    tmp_path,
    monkeypatch,
):
    first = gdprhub_case_document("gdprhub:1", "AEPD (Spain) - TEST")
    second = CaseDocument(**{**first.__dict__, "raw_text": "Different facts."})
    case_documents_path = tmp_path / "case_documents.jsonl"
    case_segments_path = tmp_path / "case_segments.jsonl"
    acquire_gdprhub_cases._write_jsonl([first, second], case_documents_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_gdprhub_cases.py",
            "--from-case-documents",
            "--case-documents",
            str(case_documents_path),
            "--case-segments",
            str(case_segments_path),
        ],
    )

    acquire_gdprhub_cases.main()

    assert len(case_segments_path.read_text(encoding="utf-8").splitlines()) == 2


def test_from_case_documents_uses_deduped_documents_for_merged_segments(
    tmp_path,
    monkeypatch,
):
    first = gdprhub_case_document("gdprhub:1", "AEPD (Spain) - TEST")
    second = CaseDocument(**{**first.__dict__, "retrieved_at": "2026-01-02T00:00:00Z"})
    case_documents_path = tmp_path / "case_documents.jsonl"
    merged_path = tmp_path / "merged.jsonl"
    case_segments_path = tmp_path / "case_segments.jsonl"
    acquire_gdprhub_cases._write_jsonl([first, second], case_documents_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_gdprhub_cases.py",
            "--from-case-documents",
            "--case-documents",
            str(case_documents_path),
            "--merged-case-documents",
            str(merged_path),
            "--case-segments",
            str(case_segments_path),
        ],
    )

    acquire_gdprhub_cases.main()

    assert acquire_gdprhub_cases.read_case_documents_jsonl(merged_path) == [first]
    assert len(case_segments_path.read_text(encoding="utf-8").splitlines()) == 1


def test_gdprhub_client_fetches_case_page_via_mediawiki_parse_api():
    session = FakeSession(
        {
            "parse": {
                "title": "AEPD (Spain) - TEST",
                "pageid": 123,
                "text": {
                    "*": (
                        "<div class='mw-parser-output'>"
                        "<table class='wikitable'>"
                        "<tr><td>Authority:</td><td>AEPD (Spain)</td></tr>"
                        "<tr><td>Jurisdiction:</td><td>Spain</td></tr>"
                        "<tr><td>Relevant Law:</td><td>Article 6 GDPR<br>Article 9 GDPR</td></tr>"
                        "<tr><td>Type:</td><td>Decision</td></tr>"
                        "<tr><td>Outcome:</td><td>Violation Found</td></tr>"
                        "<tr><td>Fine:</td><td>5000 EUR</td></tr>"
                        "<tr><td>Parties:</td><td>Retailer X</td></tr>"
                        "<tr><td>Appeal:</td><td>Unknown</td></tr>"
                        "<tr><td>Original Source:</td><td><a href='https://example.test/decision.pdf'>DPA</a></td></tr>"
                        "</table>"
                        "<h2>Facts</h2><p>The controller is a department store.</p>"
                        "</div>"
                    )
                },
                "categories": [{"*": "Article 6 GDPR"}],
                "externallinks": ["https://example.test/decision.pdf"],
            }
        }
    )
    client = GDPRhubClient(session=session)

    document = client.fetch_case_page("AEPD (Spain) - TEST")

    assert session.params["action"] == "parse"
    assert session.params["page"] == "AEPD (Spain) - TEST"
    assert document.case_id == "gdprhub:123"
    assert document.source_type == "case_summary"
    assert document.categories == ["Article 6 GDPR"]
    assert document.external_links == ["https://example.test/decision.pdf"]
    assert document.metadata["authority"] == "AEPD (Spain)"
    assert document.metadata["jurisdiction"] == "Spain"
    assert document.metadata["relevant_laws"] == ["Article 6 GDPR", "Article 9 GDPR"]
    assert document.metadata["case_type"] == "Decision"
    assert document.metadata["outcome"] == "Violation Found"
    assert document.metadata["fine"] == "5000 EUR"
    assert document.metadata["parties"] == "Retailer X"
    assert document.metadata["original_source_url"] == "https://example.test/decision.pdf"
    assert document.metadata["industry_tags"] == ["retail"]
    assert "department store" in document.raw_html
    assert "The controller is a department store." in document.raw_text


def test_gdprhub_client_retries_transient_fetch_failures():
    session = FlakySession(
        {
            "parse": {
                "title": "AEPD (Spain) - TEST",
                "pageid": 123,
                "text": {
                    "*": "<div class='mw-parser-output'><h2>Facts</h2><p>Fact text.</p></div>"
                },
                "categories": [],
                "externallinks": [],
            }
        }
    )
    client = GDPRhubClient(session=session, max_retries=1, retry_sleep_seconds=0)

    document = client.fetch_case_page("AEPD (Spain) - TEST")

    assert session.calls == 2
    assert document.case_id == "gdprhub:123"


def test_gdprhub_client_lists_all_case_titles_with_pagination_and_filtering():
    session = QueueSession(
        [
            {
                "continue": {"apcontinue": "next-title"},
                "query": {
                    "allpages": [
                        {"ns": 0, "title": "AEPD (Spain) - PS/00001/2020"},
                        {"ns": 0, "title": "Article 6 GDPR"},
                        {"ns": 0, "title": "Welcome to GDPRhub"},
                    ]
                },
            },
            {
                "query": {
                    "allpages": [
                        {"ns": 0, "title": "CNIL (France) - SAN-2021-001"},
                        {"ns": 1, "title": "Talk:AEPD (Spain) - TEST"},
                    ]
                }
            },
        ]
    )
    client = GDPRhubClient(session=session)

    titles = client.list_all_case_titles(limit=2)

    assert titles == [
        "AEPD (Spain) - PS/00001/2020",
        "CNIL (France) - SAN-2021-001",
    ]
    assert session.params_seen[0]["list"] == "allpages"
    assert session.params_seen[1]["apcontinue"] == "next-title"


def test_gdprhub_client_filters_category_members_to_case_titles():
    session = FakeSession(
        {
            "query": {
                "categorymembers": [
                    {"ns": 0, "title": "Article 6 GDPR"},
                    {"ns": 0, "title": "AEPD (Spain) - PS/00001/2020"},
                    {"ns": 1, "title": "Talk:AEPD (Spain) - TEST"},
                    {"ns": 0, "title": "Accurate titles for the decisions of DPAs"},
                ]
            }
        }
    )
    client = GDPRhubClient(session=session)

    titles = client.list_category_members("Category:Article 6 GDPR", limit=10)

    assert titles == ["AEPD (Spain) - PS/00001/2020"]
    assert session.params["list"] == "categorymembers"


def test_parse_gdprhub_case_segments_extracts_facts_and_holding_sections():
    session = FakeSession(
        {
            "parse": {
                "title": "AEPD (Spain) - TEST",
                "pageid": 123,
                "text": {
                    "*": (
                        "<div class='mw-parser-output'>"
                        "<h2><span class='mw-headline'>Facts</span></h2>"
                        "<p>The controller sent marketing emails.</p>"
                        "<h2><span class='mw-headline'>Holding</span></h2>"
                        "<p>The DPA found a violation of Article 6 GDPR.</p>"
                        "<h2><span class='mw-headline'>Comment</span></h2>"
                        "<p>Commentary is secondary.</p>"
                        "</div>"
                    )
                },
                "categories": [{"*": "Article 6 GDPR"}],
                "externallinks": [],
            }
        }
    )
    document = GDPRhubClient(session=session).fetch_case_page("AEPD (Spain) - TEST")

    segments = parse_gdprhub_case_segments(document)

    assert [segment.segment_type for segment in segments] == [
        "background",
        "reasoning",
        "comment",
    ]
    assert segments[0].heading == "Facts"
    assert segments[0].text == "The controller sent marketing emails."
    assert segments[1].heading == "Holding"
    assert "violation of Article 6" in segments[1].text
    assert segments[0].char_start >= 0
    assert segments[0].char_end > segments[0].char_start
