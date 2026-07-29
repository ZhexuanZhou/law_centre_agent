from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from legal_agentic_retrieval.evaluation import (
    BenchmarkSample,
    export_annotation_csv,
    load_benchmark,
    score_benchmark,
    score_sample,
    validate_benchmark,
)


def _sample(
    *,
    sample_id: str = "risk_001",
    relevance: list[dict] | None = None,
    coverage_groups: list[dict] | None = None,
) -> BenchmarkSample:
    return BenchmarkSample.from_mapping(
        {
            "id": sample_id,
            "split": "dev",
            "task": "risk",
            "query": "测试风险查询",
            "language": "zh-CN",
            "difficulty": "medium",
            "retrieval_k": 5,
            "relevance": relevance
            or [
                {
                    "evidence_id": "case:gdprhub:1",
                    "grade": 3,
                    "required": True,
                    "rationale": "事实相似案例",
                },
                {
                    "evidence_id": "law_unit:gdpr:article_6",
                    "grade": 2,
                    "required": True,
                    "rationale": "适用法条",
                },
            ],
            "coverage_groups": coverage_groups
            or [
                {
                    "name": "case",
                    "evidence_ids": ["case:gdprhub:1"],
                    "min_hits": 1,
                },
                {
                    "name": "law",
                    "evidence_ids": ["law_unit:gdpr:article_6"],
                    "min_hits": 1,
                },
            ],
            "expected_limitations": [],
            "tags": ["consent"],
            "annotation": {
                "status": "silver",
                "method": "evidence_anchored_manual_draft",
                "reviewer": None,
                "reviewed_at": None,
            },
        }
    )


def test_score_sample_computes_rank_and_coverage_metrics() -> None:
    sample = _sample()

    score = score_sample(
        sample,
        ["case:irrelevant", "law_unit:gdpr:article_6", "case:gdprhub:1"],
        cutoffs=(1, 2, 3),
    )

    assert score["recall@1"] == 0
    assert score["mrr@2"] == 0.5
    assert score["required_recall@2"] == 0.5
    assert score["coverage@2"] == 0.5
    assert score["recall@3"] == 1
    assert score["coverage@3"] == 1
    assert 0 < score["ndcg@3"] <= 1


def test_benchmark_rejects_unjudged_coverage_evidence() -> None:
    with pytest.raises(ValueError, match="unjudged evidence"):
        _sample(
            coverage_groups=[
                {
                    "name": "unknown",
                    "evidence_ids": ["case:not-judged"],
                    "min_hits": 1,
                }
            ]
        )


def test_benchmark_requires_singleton_coverage_evidence() -> None:
    with pytest.raises(ValueError, match="single-evidence coverage groups"):
        _sample(
            relevance=[
                {
                    "evidence_id": "case:gdprhub:1",
                    "grade": 3,
                    "required": False,
                    "rationale": "唯一覆盖该组的案例",
                }
            ],
            coverage_groups=[
                {
                    "name": "case",
                    "evidence_ids": ["case:gdprhub:1"],
                    "min_hits": 1,
                }
            ],
        )


def test_benchmark_allows_nonrequired_alternatives_in_coverage_group() -> None:
    sample = _sample(
        relevance=[
            {
                "evidence_id": "case:gdprhub:1",
                "grade": 3,
                "required": False,
                "rationale": "第一个可替代案例",
            },
            {
                "evidence_id": "case:gdprhub:2",
                "grade": 3,
                "required": False,
                "rationale": "第二个可替代案例",
            },
        ],
        coverage_groups=[
            {
                "name": "case",
                "evidence_ids": ["case:gdprhub:1", "case:gdprhub:2"],
                "min_hits": 1,
            }
        ],
    )

    assert all(not judgment.required for judgment in sample.relevance)


def test_gold_annotation_requires_reviewer_and_date() -> None:
    payload = {
        "id": "exact_law_001",
        "split": "dev",
        "task": "exact_law",
        "query": "GDPR 第六条",
        "language": "zh-CN",
        "difficulty": "easy",
        "retrieval_k": 5,
        "relevance": [
            {
                "evidence_id": "law_unit:gdpr:article_6",
                "grade": 3,
                "required": True,
                "rationale": "明确指定条款",
            }
        ],
        "coverage_groups": [],
        "expected_limitations": [],
        "tags": [],
        "annotation": {"status": "gold", "method": "human_review"},
    }

    with pytest.raises(ValueError, match="gold annotations require"):
        BenchmarkSample.from_mapping(payload)


def test_gold_annotation_rejects_invalid_review_date() -> None:
    payload = {
        "id": "exact_law_001",
        "split": "dev",
        "task": "exact_law",
        "query": "GDPR 第六条",
        "language": "zh-CN",
        "difficulty": "easy",
        "retrieval_k": 5,
        "relevance": [
            {
                "evidence_id": "law_unit:gdpr:article_6",
                "grade": 3,
                "required": True,
                "rationale": "明确指定条款",
            }
        ],
        "coverage_groups": [],
        "expected_limitations": [],
        "tags": [],
        "annotation": {
            "status": "gold",
            "method": "human_review",
            "reviewer": "reviewer",
            "reviewed_at": "2026/07/23",
        },
    }

    with pytest.raises(ValueError, match="ISO date"):
        BenchmarkSample.from_mapping(payload)


def test_validate_benchmark_checks_evidence_ids_against_index(built_index) -> None:
    index_path, _ = built_index
    valid_sample = _sample(
        relevance=[
            {
                "evidence_id": "case:gdprhub:1",
                "grade": 3,
                "required": True,
                "rationale": "索引中存在的案例",
            }
        ],
        coverage_groups=[
            {
                "name": "case",
                "evidence_ids": ["case:gdprhub:1"],
                "min_hits": 1,
            }
        ],
    )
    invalid_sample = _sample(
        sample_id="risk_002",
        relevance=[
            {
                "evidence_id": "case:missing",
                "grade": 3,
                "required": True,
                "rationale": "不存在的案例",
            }
        ],
        coverage_groups=[
            {
                "name": "case",
                "evidence_ids": ["case:missing"],
                "min_hits": 1,
            }
        ],
    )

    valid_report = validate_benchmark([valid_sample], index_path=index_path)
    invalid_report = validate_benchmark([invalid_sample], index_path=index_path)

    assert valid_report["valid"] is True
    assert invalid_report["valid"] is False
    assert invalid_report["unknown_evidence_by_sample"] == {"risk_002": ["case:missing"]}


def test_score_benchmark_reports_missing_failed_and_extra_results() -> None:
    samples = [_sample(), _sample(sample_id="risk_002")]
    results = {
        "risk_001": {
            "id": "risk_001",
            "evidence": [{"evidence_id": "case:gdprhub:1"}],
        },
        "extra": {"id": "extra", "evidence": []},
    }

    report = score_benchmark(samples, results, cutoffs=(1,))

    assert report["scored_count"] == 1
    assert report["missing_result_ids"] == ["risk_002"]
    assert report["extra_result_ids"] == ["extra"]
    assert report["overall"]["required_recall@1"] == 0.5


def test_load_benchmark_reports_line_number_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps({"not": "a sample"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark\.jsonl:1"):
        load_benchmark(path)


def test_load_results_prefers_success_over_retried_failure(tmp_path: Path) -> None:
    from legal_agentic_retrieval.evaluation import load_results

    path = tmp_path / "results.jsonl"
    records = [
        {"id": "risk_001", "evidence": [{"evidence_id": "case:1"}]},
        {"id": "risk_001", "error": {"type": "TimeoutError"}},
        {"id": "risk_002", "error": {"type": "TimeoutError"}},
        {"id": "risk_002", "evidence": [{"evidence_id": "case:2"}]},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    results = load_results(path)

    assert results["risk_001"]["evidence"][0]["evidence_id"] == "case:1"
    assert results["risk_002"]["evidence"][0]["evidence_id"] == "case:2"


def test_export_annotation_csv_contains_only_binary_review_fields(
    built_index,
    tmp_path: Path,
) -> None:
    index_path, _ = built_index
    output = tmp_path / "service.csv"

    report = export_annotation_csv(
        [_sample()],
        index_path=index_path,
        output_path=output,
    )
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert report["row_count"] == 2
    assert list(rows[0]) == [
        "sample_id",
        "task",
        "query",
        "evidence_id",
        "evidence",
        "is_relevant",
        "is_required",
    ]
    assert rows[0]["sample_id"] == "risk_001"
    assert rows[0]["task"] == "risk"
    assert rows[0]["query"] == "测试风险查询"
    assert rows[0]["evidence_id"] == "case:gdprhub:1"
    assert rows[0]["is_relevant"] == ""
    assert rows[0]["is_required"] == ""
    assert "标题：AEPD - Marketing email" in rows[0]["evidence"]
    assert "Facts:" in rows[0]["evidence"]
    assert "标题：General Data Protection Regulation" in rows[1]["evidence"]
    assert "Article 6" in rows[1]["evidence"]


def test_export_annotation_csv_can_limit_combined_evidence(
    built_index,
    tmp_path: Path,
) -> None:
    index_path, _ = built_index
    output = tmp_path / "blind.csv"

    export_annotation_csv(
        [_sample()],
        index_path=index_path,
        output_path=output,
        text_limit=40,
    )
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows[0]["evidence"]) == 40
    assert rows[0]["is_relevant"] == ""
    assert rows[0]["is_required"] == ""


def test_export_annotation_csv_rejects_missing_index_evidence(
    built_index,
    tmp_path: Path,
) -> None:
    index_path, _ = built_index
    sample = _sample(
        relevance=[
            {
                "evidence_id": "case:missing",
                "grade": 3,
                "required": True,
                "rationale": "不存在的证据",
            }
        ],
        coverage_groups=[
            {
                "name": "case",
                "evidence_ids": ["case:missing"],
                "min_hits": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="missing from the index"):
        export_annotation_csv(
            [sample],
            index_path=index_path,
            output_path=tmp_path / "missing.csv",
        )
