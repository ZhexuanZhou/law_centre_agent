from pathlib import Path
import subprocess
import sys

import crawler.law_corpus.acquire as acquire_module
import pytest
from crawler.law_corpus.acquire import acquire_sources, request_headers_for
from crawler.law_corpus.models import AcquisitionSource


class FakeResponse:
    status_code = 200
    content = b"<html><body>Article 1</body></html>"

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def get(self, url: str, timeout: int, headers=None):
        assert url == "https://example.test/gdpr"
        return FakeResponse()


class FakeBlockedResponse:
    status_code = 200
    url = "https://unblock.federalregister.gov"
    content = b"<html><title>Federal Register :: Request Access</title></html>"

    def raise_for_status(self) -> None:
        return None


class FakeBlockedSession:
    def get(self, url: str, timeout: int, headers=None):
        return FakeBlockedResponse()


class FakeEmptyResponse:
    status_code = 202
    content = b""

    def raise_for_status(self) -> None:
        return None


class FakeEmptySession:
    def get(self, url: str, timeout: int, headers=None):
        return FakeEmptyResponse()


class FakeFailingSession:
    def get(self, url: str, timeout: int, headers=None):
        raise RuntimeError("403 Client Error: Forbidden")


class FakeHeaderSession:
    def __init__(self):
        self.headers = None

    def get(self, url: str, timeout: int, headers=None):
        self.headers = headers
        return FakeResponse()


class FakePaginatedResponse:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


class FakePaginatedSession:
    def __init__(self):
        self.urls: list[str] = []

    def get(self, url: str, timeout: int, headers=None):
        self.urls.append(url)
        pages = {
            "https://example.test/law.html": (
                b"<html><body><div id='BodyLabel'>"
                b"<p>Article 1 First page text.</p>"
                b"<div id='div_currpage'><a class='page-Article' href='/law_2.html'>2</a></div>"
                b"</div><footer>site footer</footer></body></html>"
            ),
            "https://example.test/law_2.html": (
                b"<html><body><div id='content'>"
                b"<p>Article 2 Second page text.</p>"
                b"<div id='div_currpage'>previous</div>"
                b"</div><footer>site footer</footer></body></html>"
            ),
        }
        return FakePaginatedResponse(pages[url])


class FakeIframeSession:
    def __init__(self):
        self.urls: list[str] = []

    def get(self, url: str, timeout: int, headers=None):
        self.urls.append(url)
        pages = {
            "https://example.test/latest/text": (
                b"<html><body><iframe id='epubFrame' "
                b"src='/current/text/original/epub/OEBPS/document_1/document_1.html'>"
                b"</iframe><nav>table of contents</nav></body></html>"
            ),
            "https://example.test/current/text/original/epub/OEBPS/document_1/document_1.html": (
                b"<html><body><h1>Privacy Act 1988</h1>"
                b"<h2>1 Short title</h2><p>This Act is the Privacy Act 1988.</p>"
                b"</body></html>"
            ),
        }
        return FakePaginatedResponse(pages[url])


def test_acquire_auto_source_writes_file_and_metadata(tmp_path: Path):
    source = AcquisitionSource(
        doc_id="eu_gdpr_2016_679",
        title="GDPR",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        language="en",
        url="https://example.test/gdpr",
        preferred_format="html",
        download_mode="auto",
        target_path=str(tmp_path / "data/raw/laws/EU/eu_gdpr/2016-04-27/eu_gdpr_2016_679.html"),
    )

    results = acquire_sources(
        [source], session=FakeSession(), manual_report_path=tmp_path / "manual.md"
    )

    assert results[0].status == "downloaded"
    assert Path(source.target_path).exists()
    assert Path(results[0].metadata_path).exists()


def test_acquire_auto_html_source_combines_paginated_article_body(tmp_path: Path):
    target_path = tmp_path / "data/raw/laws/CN/china_csl/china_csl.html"
    session = FakePaginatedSession()
    source = AcquisitionSource(
        doc_id="china_cybersecurity_law_2016",
        title="中华人民共和国网络安全法",
        jurisdiction="CN",
        law_family="china_csl",
        source_type="primary_law",
        version_date="2016-11-07",
        effective_date="2017-06-01",
        language="zh",
        url="https://example.test/law.html",
        preferred_format="html",
        download_mode="auto",
        target_path=str(target_path),
    )

    results = acquire_sources([source], session=session, manual_report_path=tmp_path / "manual.md")

    assert results[0].status == "downloaded"
    assert session.urls == ["https://example.test/law.html", "https://example.test/law_2.html"]
    html = target_path.read_text(encoding="utf-8")
    assert "Article 1 First page text." in html
    assert "Article 2 Second page text." in html
    assert "div_currpage" not in html
    assert "site footer" not in html


def test_acquire_auto_html_source_follows_epub_iframe(tmp_path: Path):
    target_path = tmp_path / "data/raw/laws/AU/privacy_act/privacy_act.html"
    session = FakeIframeSession()
    source = AcquisitionSource(
        doc_id="australia_privacy_act_1988_current",
        title="Privacy Act 1988",
        jurisdiction="AU",
        law_family="australia_privacy_act",
        source_type="primary_law",
        version_date="current",
        effective_date="",
        language="en",
        url="https://example.test/latest/text",
        preferred_format="html",
        download_mode="auto",
        target_path=str(target_path),
    )

    results = acquire_sources([source], session=session, manual_report_path=tmp_path / "manual.md")

    assert results[0].status == "downloaded"
    assert session.urls == [
        "https://example.test/latest/text",
        "https://example.test/current/text/original/epub/OEBPS/document_1/document_1.html",
    ]
    html = target_path.read_text(encoding="utf-8")
    assert "This Act is the Privacy Act 1988." in html
    assert "table of contents" not in html


def test_acquire_auto_source_uses_curl_fallback_for_default_session_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target_path = tmp_path / "data/raw/laws/SG/singapore_pdpa/current/singapore_pdpa_2012.pdf"
    source = AcquisitionSource(
        doc_id="singapore_pdpa_2012",
        title="Personal Data Protection Act 2012",
        jurisdiction="SG",
        law_family="singapore_pdpa",
        source_type="primary_law",
        version_date="current",
        effective_date="",
        language="en",
        url="https://sso.agc.gov.sg/Act/PDPA2012?ViewType=Pdf",
        preferred_format="pdf",
        download_mode="auto",
        target_path=str(target_path),
    )
    commands: list[list[str]] = []

    def fake_run(command, check, capture_output):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"%PDF-1.7\nSingapore PDPA")

    monkeypatch.setattr(acquire_module.requests, "Session", lambda: FakeFailingSession())
    monkeypatch.setattr(acquire_module.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(acquire_module.subprocess, "run", fake_run)

    results = acquire_sources([source], manual_report_path=tmp_path / "manual.md")

    assert results[0].status == "downloaded"
    assert "curl fallback" in results[0].message
    assert target_path.read_bytes().startswith(b"%PDF-1.7")
    assert "--location" in commands[0]
    assert "Accept: application/pdf,text/html,application/xhtml+xml,*/*;q=0.8" in commands[0]


def test_acquire_xml_source_requests_xml_notice_tree(tmp_path: Path):
    session = FakeHeaderSession()
    source = AcquisitionSource(
        doc_id="eu_gdpr_2016_679",
        title="GDPR",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        language="en",
        url="https://publications.europa.eu/resource/celex/32016R0679",
        preferred_format="xml",
        download_mode="auto",
        target_path=str(tmp_path / "data/raw/laws/EU/eu_gdpr/2016-04-27/eu_gdpr_2016_679.xml"),
    )

    acquire_sources([source], session=session, manual_report_path=tmp_path / "manual.md")

    assert session.headers["Accept"] == "application/xml;notice=tree"
    assert "law-centre-crawler" in session.headers["User-Agent"]


def test_acquire_pdf_source_requests_pdf_content():
    source = AcquisitionSource(
        doc_id="singapore_pdpa_2012",
        title="Personal Data Protection Act 2012",
        jurisdiction="SG",
        law_family="singapore_pdpa",
        source_type="primary_law",
        version_date="current",
        effective_date="",
        language="en",
        url="https://sso.agc.gov.sg/Act/PDPA2012?ViewType=Pdf",
        preferred_format="pdf",
        download_mode="auto",
        target_path="corpus/raw/laws/SG/singapore_pdpa/current/singapore_pdpa_2012.pdf",
    )

    headers = request_headers_for(source)

    assert headers["Accept"].startswith("application/pdf")
    assert "text/html" in headers["Accept"]
    assert "law-centre-crawler" not in headers["User-Agent"]


def test_acquire_no_manual_report_ends_with_newline(tmp_path: Path):
    source = AcquisitionSource(
        doc_id="eu_gdpr_2016_679",
        title="GDPR",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        language="en",
        url="https://example.test/gdpr",
        preferred_format="html",
        download_mode="auto",
        target_path=str(tmp_path / "data/raw/laws/EU/eu_gdpr/2016-04-27/eu_gdpr_2016_679.html"),
    )

    acquire_sources([source], session=FakeSession(), manual_report_path=tmp_path / "manual.md")

    assert (tmp_path / "manual.md").read_text(encoding="utf-8").endswith("\n")


def test_acquire_blocked_access_page_is_failed_and_reported(tmp_path: Path):
    target_path = tmp_path / "data/raw/laws/US/us_cfr/2026-06-04/us_coppa_16_cfr_part_312.xml"
    source = AcquisitionSource(
        doc_id="us_coppa_16_cfr_part_312",
        title="16 CFR Part 312 Children's Online Privacy Protection Rule",
        jurisdiction="US",
        law_family="us_cfr",
        source_type="primary_law",
        version_date="2026-06-04",
        effective_date="",
        language="en",
        url="https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-312",
        preferred_format="xml",
        download_mode="auto",
        target_path=str(target_path),
    )

    results = acquire_sources(
        [source], session=FakeBlockedSession(), manual_report_path=tmp_path / "manual.md"
    )

    assert results[0].status == "failed"
    assert not target_path.exists()
    report = (tmp_path / "manual.md").read_text(encoding="utf-8")
    assert "us_coppa_16_cfr_part_312" in report


def test_acquire_empty_response_is_failed_and_reported(tmp_path: Path):
    target_path = tmp_path / "data/raw/laws/EU/eu_gdpr/2016-04-27/eu_gdpr_2016_679.html"
    source = AcquisitionSource(
        doc_id="eu_gdpr_2016_679",
        title="GDPR",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        language="en",
        url="https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng/",
        preferred_format="html",
        download_mode="auto",
        target_path=str(target_path),
        manual_instructions="Open the official EUR-Lex page and save the legal text.",
    )

    results = acquire_sources(
        [source], session=FakeEmptySession(), manual_report_path=tmp_path / "manual.md"
    )

    assert results[0].status == "failed"
    assert not target_path.exists()
    report = (tmp_path / "manual.md").read_text(encoding="utf-8")
    assert "eu_gdpr_2016_679" in report
    assert "EUR-Lex" in report


def test_acquire_manual_source_writes_manual_report(tmp_path: Path):
    source = AcquisitionSource(
        doc_id="hong_kong_pdpo_cap486",
        title="PDPO",
        jurisdiction="HK",
        law_family="hong_kong_pdpo",
        source_type="primary_law",
        version_date="current",
        effective_date="",
        language="en-zh",
        url="https://example.test/pdpo",
        preferred_format="txt",
        download_mode="manual",
        target_path=str(
            tmp_path / "data/raw/laws/HK/hong_kong_pdpo/current/hong_kong_pdpo_cap486.txt"
        ),
        manual_instructions="Download and save the text.",
    )

    results = acquire_sources(
        [source], session=FakeSession(), manual_report_path=tmp_path / "manual.md"
    )

    assert results[0].status == "manual_required"
    report = (tmp_path / "manual.md").read_text(encoding="utf-8")
    assert "hong_kong_pdpo_cap486" in report
    assert source.target_path in report
    assert report.endswith("\n")


def test_acquire_script_runs_from_repo_root_without_editable_install(tmp_path: Path):
    catalog = tmp_path / "laws.seed.toml"
    target_path = tmp_path / "data/raw/laws/HK/hong_kong_pdpo/current/hong_kong_pdpo_cap486.txt"
    manual_report = tmp_path / "manual.md"
    catalog.write_text(
        f"""
[[sources]]
doc_id = "hong_kong_pdpo_cap486"
title = "PDPO"
jurisdiction = "HK"
law_family = "hong_kong_pdpo"
source_type = "primary_law"
version_date = "current"
effective_date = ""
language = "en-zh"
url = "https://example.test/pdpo"
preferred_format = "txt"
download_mode = "manual"
target_path = "{target_path}"
manual_instructions = "Download and save the text."
""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "tools/acquire_law_sources.py",
            "--catalog",
            str(catalog),
            "--manual-report",
            str(manual_report),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "hong_kong_pdpo_cap486\tmanual_required" in result.stdout
    assert manual_report.exists()
