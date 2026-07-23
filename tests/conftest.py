from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pytest

from legal_agentic_retrieval.index import CorpusIndexBuilder


class FakeEmbedder:
    model_name = "test-embedding"
    dimension = 32
    batch_size = 8
    max_input_chars = 8192

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w\u3400-\u9fff]+", text.casefold())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                matrix[row, int.from_bytes(digest[:2], "big") % self.dimension] += 1
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def built_index(tmp_path: Path) -> tuple[Path, FakeEmbedder]:
    corpus = tmp_path / "structured"
    corpus.mkdir()
    _write_jsonl(
        corpus / "laws.jsonl",
        [
            {
                "doc_id": "eu_gdpr_2016_679",
                "title": "General Data Protection Regulation",
                "jurisdiction": "EU",
                "language": "en",
                "law_family": "privacy",
                "source_url": "https://example.test/gdpr",
                "version_date": "2025-01-01",
                "effective_date": "2018-05-25",
                "raw_text": "not copied",
            },
            {
                "doc_id": "china_pipl_2021",
                "title": "中华人民共和国个人信息保护法",
                "jurisdiction": "CN",
                "language": "zh",
                "law_family": "privacy",
                "source_url": "https://example.test/pipl",
                "version_date": "2021-11-01",
                "effective_date": "2021-11-01",
                "raw_text": "not copied",
            },
        ],
    )
    _write_jsonl(
        corpus / "legal_units.jsonl",
        [
            _unit(
                "gdpr:article_6",
                "eu_gdpr_2016_679",
                "EU",
                "Article 6",
                " ".join(["Lawful processing includes consent and other legal bases."] * 35),
            ),
            _unit(
                "pipl:article_13",
                "china_pipl_2021",
                "CN",
                "Article 13",
                "处理个人信息应当取得个人同意或者具有其他合法性基础。",
            ),
            {
                **_unit("gdpr:old", "eu_gdpr_2016_679", "EU", "Article 99", "Old text."),
                "is_current": False,
            },
        ],
    )
    _write_jsonl(
        corpus / "gdprhub_cases.jsonl",
        [
            {
                "case_id": "gdprhub:1",
                "title": "AEPD - Marketing email",
                "authority": "AEPD",
                "jurisdiction": "ES",
                "country": "Spain",
                "decided_date": "2025-02-01",
                "case_number": "PS/1/2025",
                "ecli": None,
                "company_or_parties": "Example Shop",
                "facts_text": " ".join(
                    ["The shop sent direct marketing email without consent."] * 8
                ),
                "decision_text": " ".join(
                    ["The authority found that valid consent was absent."] * 8
                ),
                "outcome": "violation",
                "source_url": "https://gdprhub.eu/1",
                "original_source_url": "https://authority.test/1",
                "industry": [{"industry": "retail_ecommerce"}],
                "categories": ["Article_6_GDPR"],
            }
        ],
    )
    _write_jsonl(
        corpus / "case_law_relations.jsonl",
        [
            {
                "case_id": "gdprhub:1",
                "target_unit_id": "gdpr:article_6",
                "citation": "Article 6 GDPR",
                "resolution_status": "resolved",
            }
        ],
    )
    _write_jsonl(corpus / "law_relations.jsonl", [])
    embedder = FakeEmbedder()
    index_path = tmp_path / "index.sqlite3"
    CorpusIndexBuilder(
        corpus,
        embedder,
        passage_threshold_tokens=40,
        passage_target_tokens=20,
        passage_max_tokens=30,
        passage_overlap_tokens=5,
    ).build(index_path)
    return index_path, embedder


def _unit(
    unit_id: str,
    doc_id: str,
    jurisdiction: str,
    citation: str,
    text: str,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "source_doc_id": doc_id,
        "parent_id": None,
        "unit_type": "article",
        "canonical_citation": f"{doc_id} {citation}",
        "local_citation": citation,
        "text": text,
        "jurisdiction": jurisdiction,
        "law_name": doc_id,
        "effective_from": None,
        "effective_to": None,
        "is_current": True,
    }
