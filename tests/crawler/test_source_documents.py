import json
import subprocess
import sys
from pathlib import Path

from crawler.law_corpus.models import AcquisitionSource, SourceDocument
from crawler.law_corpus.source_documents import (
    DuplicateSourceDocumentError,
    build_source_document,
    build_source_documents_from_catalogs,
    dedupe_source_documents,
    drop_superseded_source_documents,
)


def write_catalog(path: Path, sources: list[dict[str, str]]) -> None:
    sections: list[str] = []
    for source in sources:
        sections.append(
            "\n".join(
                [
                    "[[sources]]",
                    f'doc_id = "{source["doc_id"]}"',
                    f'title = "{source["title"]}"',
                    f'jurisdiction = "{source["jurisdiction"]}"',
                    f'law_family = "{source["law_family"]}"',
                    f'source_type = "{source["source_type"]}"',
                    f'version_date = "{source["version_date"]}"',
                    f'effective_date = "{source["effective_date"]}"',
                    f'language = "{source["language"]}"',
                    f'url = "{source["url"]}"',
                    f'preferred_format = "{source["preferred_format"]}"',
                    f'download_mode = "{source["download_mode"]}"',
                    f'target_path = "{source["target_path"]}"',
                    f'source_set = "{source["source_set"]}"',
                    f'translation_status = "{source["translation_status"]}"',
                ]
            )
        )
    path.write_text("\n\n".join(sections), encoding="utf-8")


def catalog_source(
    *,
    doc_id: str,
    title: str,
    target_path: Path,
    source_set: str,
    url: str,
) -> dict[str, str]:
    return {
        "doc_id": doc_id,
        "title": title,
        "jurisdiction": "UK",
        "law_family": "uk_dpa",
        "source_type": "primary_law",
        "version_date": "current",
        "effective_date": "",
        "language": "en",
        "url": url,
        "preferred_format": "txt",
        "download_mode": "auto",
        "target_path": str(target_path),
        "source_set": source_set,
        "translation_status": "official_original",
    }


def test_build_source_document_from_raw_file_and_metadata(tmp_path: Path):
    raw_path = tmp_path / "pipl.txt"
    raw_path.write_text("第一条 为了保护个人信息权益。", encoding="utf-8")

    source = AcquisitionSource(
        doc_id="china_pipl_2021",
        title="中华人民共和国个人信息保护法",
        jurisdiction="CN",
        law_family="china_pipl",
        source_type="primary_law",
        version_date="2021-08-20",
        effective_date="2021-11-01",
        language="zh",
        url="https://example.test/pipl",
        preferred_format="txt",
        download_mode="auto",
        target_path=str(raw_path),
    )

    doc = build_source_document(source)

    assert doc.doc_id == "china_pipl_2021"
    assert doc.raw_text == "第一条 为了保护个人信息权益。"
    assert json.loads(doc.to_json())["jurisdiction"] == "CN"


def test_build_source_document_preserves_source_package_metadata(tmp_path: Path):
    raw_path = tmp_path / "uk_dpa.xml"
    raw_path.write_text(
        "<Act><Title>Data Protection Act 2018</Title><P1>1 Overview</P1></Act>", encoding="utf-8"
    )
    source = AcquisitionSource(
        doc_id="uk_data_protection_act_2018",
        title="Data Protection Act 2018",
        jurisdiction="UK",
        law_family="uk_dpa",
        source_type="primary_law",
        version_date="current",
        effective_date="",
        language="en",
        url="https://www.legislation.gov.uk/ukpga/2018/12/data.xml",
        preferred_format="xml",
        download_mode="auto",
        target_path=str(raw_path),
        source_set="expansion_major_markets",
        translation_status="official_original",
    )

    document = build_source_document(source)

    assert document.metadata["source_set"] == "expansion_major_markets"
    assert document.metadata["translation_status"] == "official_original"
    assert document.metadata["acquisition_mode"] == "auto"


def test_dedupe_source_documents_keeps_identical_duplicate_doc_id():
    document = SourceDocument(
        doc_id="uk_data_protection_act_2018",
        jurisdiction="UK",
        law_family="uk_dpa",
        source_type="primary_law",
        title="Data Protection Act 2018",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/uk-dpa",
        language="en",
        raw_text="1 Overview",
        metadata={"source_set": "expansion_major_markets"},
    )

    assert dedupe_source_documents([document, document]) == [document]


def test_dedupe_source_documents_rejects_conflicting_duplicate_doc_id():
    first = SourceDocument(
        doc_id="uk_data_protection_act_2018",
        jurisdiction="UK",
        law_family="uk_dpa",
        source_type="primary_law",
        title="Data Protection Act 2018",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/uk-dpa",
        language="en",
        raw_text="1 Overview",
        metadata={"source_set": "seed", "target_path": "/tmp/seed.txt"},
    )
    second = SourceDocument(
        **{
            **first.__dict__,
            "source_url": "https://example.test/uk-dpa-expansion",
            "raw_text": "2 Different text",
            "metadata": {
                "source_set": "expansion_major_markets",
                "target_path": "/tmp/expansion.txt",
            },
        }
    )

    try:
        dedupe_source_documents([first, second])
    except DuplicateSourceDocumentError as exc:
        message = str(exc)
        assert "doc_id=uk_data_protection_act_2018" in message
        assert "metadata" in message
        assert "raw_text" in message
        assert "source_url" in message
        assert "existing(metadata.source_set=seed" in message
        assert "metadata.target_path=/tmp/seed.txt" in message
        assert "source_url=https://example.test/uk-dpa" in message
        assert "incoming(metadata.source_set=expansion_major_markets" in message
        assert "metadata.target_path=/tmp/expansion.txt" in message
        assert "source_url=https://example.test/uk-dpa-expansion" in message
    else:
        raise AssertionError("Expected DuplicateSourceDocumentError")


def test_drop_superseded_source_documents_removes_ccpa_seed_when_complete_source_exists():
    seed_document = SourceDocument(
        doc_id="us_ccpa_cpra_1798_100",
        jurisdiction="US-CA",
        law_family="us_state_privacy",
        source_type="primary_law",
        title="California Consumer Privacy Act and CPRA Amendments",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/seed-ccpa",
        language="en",
        raw_text="1798.100. A consumer shall have the right...",
        metadata={"source_set": "seed"},
    )
    complete_document = SourceDocument(
        doc_id="us_ca_ccpa_cpra_civ_1798_100_199",
        jurisdiction="US-CA",
        law_family="us_state_privacy",
        source_type="primary_law",
        title="California Consumer Privacy Act and CPRA Amendments",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/full-ccpa",
        language="en",
        raw_text="1798.100. A consumer...\n1798.105. A consumer...",
        metadata={"source_set": "expansion_major_markets"},
    )

    assert drop_superseded_source_documents([seed_document, complete_document]) == [
        complete_document
    ]
    assert drop_superseded_source_documents([seed_document]) == [seed_document]


def test_build_source_documents_from_catalogs_preserves_order_and_dedupes_identical_docs(
    tmp_path: Path,
):
    alpha_raw = tmp_path / "alpha.txt"
    beta_raw = tmp_path / "beta.txt"
    alpha_raw.write_text("Alpha section", encoding="utf-8")
    beta_raw.write_text("Beta section", encoding="utf-8")

    alpha_source = catalog_source(
        doc_id="uk_alpha_act",
        title="Alpha Act",
        target_path=alpha_raw,
        source_set="seed",
        url="https://example.test/alpha",
    )
    beta_source = catalog_source(
        doc_id="uk_beta_act",
        title="Beta Act",
        target_path=beta_raw,
        source_set="expansion_major_markets",
        url="https://example.test/beta",
    )

    first_catalog = tmp_path / "first.toml"
    second_catalog = tmp_path / "second.toml"
    write_catalog(first_catalog, [alpha_source])
    write_catalog(second_catalog, [alpha_source, beta_source])

    documents = build_source_documents_from_catalogs([first_catalog, second_catalog])

    assert [document.doc_id for document in documents] == ["uk_alpha_act", "uk_beta_act"]
    assert [document.raw_text for document in documents] == ["Alpha section", "Beta section"]


def test_build_source_documents_cli_requires_out_for_expansion_catalog(tmp_path: Path):
    raw_path = tmp_path / "alpha.txt"
    raw_path.write_text("Alpha section", encoding="utf-8")
    catalog_path = tmp_path / "laws.expansion.toml"
    write_catalog(
        catalog_path,
        [
            catalog_source(
                doc_id="uk_alpha_act",
                title="Alpha Act",
                target_path=raw_path,
                source_set="expansion_major_markets",
                url="https://example.test/alpha",
            )
        ],
    )
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "build_source_documents.py"),
            "--catalog",
            str(catalog_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--out is required" in result.stderr
