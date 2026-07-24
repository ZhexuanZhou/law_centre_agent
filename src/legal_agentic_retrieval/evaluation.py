from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import sqlite3
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from legal_agentic_retrieval.models import TaskMode


ALLOWED_TASKS = {"exact_law", "risk", "compare", "case_search"}
ALLOWED_ANNOTATION_STATUSES = {"draft", "silver", "gold"}
ANNOTATION_CSV_COLUMNS = (
    "sample_id",
    "task",
    "query",
    "evidence_id",
    "evidence",
    "is_relevant",
    "is_required",
)


@dataclass(frozen=True)
class RelevanceJudgment:
    evidence_id: str
    grade: int
    required: bool
    rationale: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RelevanceJudgment:
        evidence_id = str(value.get("evidence_id") or "").strip()
        grade = value.get("grade")
        rationale = str(value.get("rationale") or "").strip()
        if not evidence_id:
            raise ValueError("relevance evidence_id must not be empty")
        if not isinstance(grade, int) or isinstance(grade, bool) or grade not in {1, 2, 3}:
            raise ValueError(f"relevance grade must be 1, 2, or 3 for {evidence_id}")
        if not rationale:
            raise ValueError(f"relevance rationale must not be empty for {evidence_id}")
        return cls(
            evidence_id=evidence_id,
            grade=grade,
            required=bool(value.get("required", False)),
            rationale=rationale,
        )


@dataclass(frozen=True)
class CoverageGroup:
    name: str
    evidence_ids: tuple[str, ...]
    min_hits: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CoverageGroup:
        name = str(value.get("name") or "").strip()
        evidence_ids = _unique_strings(value.get("evidence_ids"))
        min_hits = value.get("min_hits", 1)
        if not name:
            raise ValueError("coverage group name must not be empty")
        if not evidence_ids:
            raise ValueError(f"coverage group {name!r} must contain evidence_ids")
        if (
            not isinstance(min_hits, int)
            or isinstance(min_hits, bool)
            or min_hits < 1
            or min_hits > len(evidence_ids)
        ):
            raise ValueError(
                f"coverage group {name!r} min_hits must be between 1 and {len(evidence_ids)}"
            )
        return cls(name=name, evidence_ids=evidence_ids, min_hits=min_hits)


@dataclass(frozen=True)
class Annotation:
    status: str
    method: str
    reviewer: str | None = None
    reviewed_at: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Annotation:
        status = str(value.get("status") or "").strip()
        method = str(value.get("method") or "").strip()
        reviewer = _optional_string(value.get("reviewer"))
        reviewed_at = _optional_string(value.get("reviewed_at"))
        if status not in ALLOWED_ANNOTATION_STATUSES:
            raise ValueError(
                f"annotation status must be one of {sorted(ALLOWED_ANNOTATION_STATUSES)}"
            )
        if not method:
            raise ValueError("annotation method must not be empty")
        if status == "gold" and (not reviewer or not reviewed_at):
            raise ValueError("gold annotations require reviewer and reviewed_at")
        if status == "gold":
            try:
                date.fromisoformat(reviewed_at)
            except ValueError as exc:
                raise ValueError(
                    "gold annotation reviewed_at must be an ISO date (YYYY-MM-DD)"
                ) from exc
        return cls(
            status=status,
            method=method,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
        )


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    split: str
    task: TaskMode
    query: str
    language: str
    difficulty: str
    retrieval_k: int
    relevance: tuple[RelevanceJudgment, ...]
    coverage_groups: tuple[CoverageGroup, ...]
    expected_limitations: tuple[str, ...]
    tags: tuple[str, ...]
    annotation: Annotation

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkSample:
        sample_id = str(value.get("id") or "").strip()
        split = str(value.get("split") or "").strip()
        task = str(value.get("task") or "").strip()
        query = str(value.get("query") or "").strip()
        language = str(value.get("language") or "").strip()
        difficulty = str(value.get("difficulty") or "").strip()
        retrieval_k = value.get("retrieval_k", 10)
        if not sample_id:
            raise ValueError("sample id must not be empty")
        if split not in {"dev", "test"}:
            raise ValueError(f"split must be dev or test for sample {sample_id}")
        if task not in ALLOWED_TASKS:
            raise ValueError(f"unsupported task {task!r} for sample {sample_id}")
        if not query:
            raise ValueError(f"query must not be empty for sample {sample_id}")
        if not language:
            raise ValueError(f"language must not be empty for sample {sample_id}")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"invalid difficulty for sample {sample_id}: {difficulty!r}")
        if (
            not isinstance(retrieval_k, int)
            or isinstance(retrieval_k, bool)
            or not 1 <= retrieval_k <= 50
        ):
            raise ValueError(f"retrieval_k must be between 1 and 50 for sample {sample_id}")

        raw_relevance = value.get("relevance")
        if not isinstance(raw_relevance, list) or not raw_relevance:
            raise ValueError(f"relevance must be a non-empty list for sample {sample_id}")
        relevance = tuple(
            RelevanceJudgment.from_mapping(item)
            for item in raw_relevance
            if isinstance(item, Mapping)
        )
        if len(relevance) != len(raw_relevance):
            raise ValueError(f"every relevance item must be an object for sample {sample_id}")
        duplicate_evidence = _duplicates(item.evidence_id for item in relevance)
        if duplicate_evidence:
            raise ValueError(
                f"duplicate relevance evidence for sample {sample_id}: "
                + ", ".join(duplicate_evidence)
            )

        raw_groups = value.get("coverage_groups") or []
        if not isinstance(raw_groups, list):
            raise ValueError(f"coverage_groups must be a list for sample {sample_id}")
        coverage_groups = tuple(
            CoverageGroup.from_mapping(item) for item in raw_groups if isinstance(item, Mapping)
        )
        if len(coverage_groups) != len(raw_groups):
            raise ValueError(f"every coverage group must be an object for sample {sample_id}")
        duplicate_groups = _duplicates(group.name for group in coverage_groups)
        if duplicate_groups:
            raise ValueError(
                f"duplicate coverage group names for sample {sample_id}: "
                + ", ".join(duplicate_groups)
            )
        judged_ids = {item.evidence_id for item in relevance}
        unjudged_group_ids = sorted(
            {
                evidence_id
                for group in coverage_groups
                for evidence_id in group.evidence_ids
                if evidence_id not in judged_ids
            }
        )
        if unjudged_group_ids:
            raise ValueError(
                f"coverage groups reference unjudged evidence for sample {sample_id}: "
                + ", ".join(unjudged_group_ids)
            )

        annotation_value = value.get("annotation")
        if not isinstance(annotation_value, Mapping):
            raise ValueError(f"annotation must be an object for sample {sample_id}")
        return cls(
            sample_id=sample_id,
            split=split,
            task=task,  # type: ignore[arg-type]
            query=query,
            language=language,
            difficulty=difficulty,
            retrieval_k=retrieval_k,
            relevance=relevance,
            coverage_groups=coverage_groups,
            expected_limitations=_unique_strings(value.get("expected_limitations")),
            tags=_unique_strings(value.get("tags")),
            annotation=Annotation.from_mapping(annotation_value),
        )


def load_benchmark(path: str | Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("sample must be a JSON object")
                samples.append(BenchmarkSample.from_mapping(value))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid benchmark sample at {path}:{line_number}: {exc}"
                ) from exc
    if not samples:
        raise ValueError(f"benchmark contains no samples: {path}")
    duplicate_ids = _duplicates(sample.sample_id for sample in samples)
    if duplicate_ids:
        raise ValueError("duplicate benchmark sample ids: " + ", ".join(duplicate_ids))
    return samples


def validate_benchmark(
    samples: Sequence[BenchmarkSample],
    *,
    index_path: str | Path,
) -> dict[str, Any]:
    known_evidence = evidence_ids_in_index(index_path)
    unknown_by_sample: dict[str, list[str]] = {}
    task_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for sample in samples:
        task_counts[sample.task] = task_counts.get(sample.task, 0) + 1
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        status = sample.annotation.status
        status_counts[status] = status_counts.get(status, 0) + 1
        unknown = sorted(
            {
                judgment.evidence_id
                for judgment in sample.relevance
                if judgment.evidence_id not in known_evidence
            }
        )
        if unknown:
            unknown_by_sample[sample.sample_id] = unknown
    return {
        "valid": not unknown_by_sample,
        "sample_count": len(samples),
        "task_counts": dict(sorted(task_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "known_index_evidence_count": len(known_evidence),
        "unknown_evidence_by_sample": unknown_by_sample,
    }


def evidence_ids_in_index(index_path: str | Path) -> set[str]:
    with sqlite3.connect(index_path) as connection:
        law_ids = {
            f"law_unit:{row[0]}" for row in connection.execute("SELECT unit_id FROM law_units")
        }
        case_ids = {f"case:{row[0]}" for row in connection.execute("SELECT case_id FROM cases")}
    return law_ids | case_ids


def export_annotation_csv(
    samples: Sequence[BenchmarkSample],
    *,
    index_path: str | Path,
    output_path: str | Path,
    text_limit: int = 0,
) -> dict[str, Any]:
    """Export the minimum query-evidence fields needed for binary legal review."""
    if text_limit < 0:
        raise ValueError("text_limit must be zero or a positive integer")
    if not samples:
        raise ValueError("at least one benchmark sample is required")

    requested_evidence_ids = {
        judgment.evidence_id for sample in samples for judgment in sample.relevance
    }
    evidence_by_id = _load_evidence_records(index_path, requested_evidence_ids)
    missing_evidence_ids = sorted(requested_evidence_ids - set(evidence_by_id))
    if missing_evidence_ids:
        raise ValueError(
            "benchmark evidence is missing from the index: " + ", ".join(missing_evidence_ids)
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    task_counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ANNOTATION_CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for sample in samples:
            task_counts[sample.task] = task_counts.get(sample.task, 0) + 1
            for judgment in sample.relevance:
                evidence = evidence_by_id[judgment.evidence_id]
                review_text = _format_review_evidence(evidence)
                review_text, _ = _limit_text(review_text, text_limit)
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "query": sample.query,
                        "evidence_id": judgment.evidence_id,
                        "evidence": review_text,
                        "is_relevant": "",
                        "is_required": "",
                    }
                )
                row_count += 1
    return {
        "output": str(output),
        "sample_count": len(samples),
        "row_count": row_count,
        "task_counts": dict(sorted(task_counts.items())),
        "text_limit": text_limit,
    }


def _load_evidence_records(
    index_path: str | Path,
    evidence_ids: set[str],
) -> dict[str, dict[str, Any]]:
    law_unit_ids = sorted(
        evidence_id.removeprefix("law_unit:")
        for evidence_id in evidence_ids
        if evidence_id.startswith("law_unit:")
    )
    case_ids = sorted(
        evidence_id.removeprefix("case:")
        for evidence_id in evidence_ids
        if evidence_id.startswith("case:")
    )
    records: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(index_path) as connection:
        connection.row_factory = sqlite3.Row
        for row in _select_in_batches(
            connection,
            """
            SELECT
                u.unit_id, u.unit_type, u.canonical_citation, u.local_citation,
                u.text, u.jurisdiction, u.effective_from, u.effective_to,
                l.title, l.source_url
            FROM law_units AS u
            JOIN laws AS l ON l.doc_id = u.doc_id
            WHERE u.unit_id IN ({placeholders})
            """,
            law_unit_ids,
        ):
            records[f"law_unit:{row['unit_id']}"] = {
                "evidence_id": f"law_unit:{row['unit_id']}",
                "source_type": "law_unit",
                "title": row["title"] or "",
                "citation": row["canonical_citation"] or row["local_citation"] or "",
                "jurisdiction": row["jurisdiction"] or "",
                "country": "",
                "authority": "",
                "decided_date": "",
                "outcome": "",
                "unit_type": row["unit_type"] or "",
                "effective_from": row["effective_from"] or "",
                "effective_to": row["effective_to"] or "",
                "source_url": row["source_url"] or "",
                "evidence_text": row["text"] or "",
            }
        for row in _select_in_batches(
            connection,
            """
            SELECT
                case_id, title, authority, jurisdiction, country, decided_date,
                case_number, ecli, facts_text, decision_text, outcome, source_url
            FROM cases
            WHERE case_id IN ({placeholders})
            """,
            case_ids,
        ):
            citation = row["ecli"] or row["case_number"] or row["case_id"]
            facts = str(row["facts_text"] or "").strip()
            decision = str(row["decision_text"] or "").strip()
            records[f"case:{row['case_id']}"] = {
                "evidence_id": f"case:{row['case_id']}",
                "source_type": "case",
                "title": row["title"] or "",
                "citation": citation,
                "jurisdiction": row["jurisdiction"] or "",
                "country": row["country"] or "",
                "authority": row["authority"] or "",
                "decided_date": row["decided_date"] or "",
                "outcome": row["outcome"] or "",
                "unit_type": "",
                "effective_from": "",
                "effective_to": "",
                "source_url": row["source_url"] or "",
                "evidence_text": f"Facts: {facts}\nDecision: {decision}".strip(),
            }
    return records


def _select_in_batches(
    connection: sqlite3.Connection,
    query_template: str,
    values: Sequence[str],
    *,
    batch_size: int = 500,
) -> Iterable[sqlite3.Row]:
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        yield from connection.execute(
            query_template.format(placeholders=placeholders),
            batch,
        )


def _format_review_evidence(evidence: Mapping[str, Any]) -> str:
    header = [
        f"标题：{evidence['title']}",
        f"引用：{evidence['citation']}",
        f"法域：{evidence['jurisdiction']}",
    ]
    if evidence["country"]:
        header.append(f"国家：{evidence['country']}")
    if evidence["authority"]:
        header.append(f"机构：{evidence['authority']}")
    if evidence["decided_date"]:
        header.append(f"日期：{evidence['decided_date']}")
    return "\n".join([*header, "", str(evidence["evidence_text"])])


def _limit_text(text: str, text_limit: int) -> tuple[str, bool]:
    if not text_limit or len(text) <= text_limit:
        return text, False
    return text[:text_limit], True


def load_results(path: str | Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid result JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"result must be a JSON object at {path}:{line_number}")
            sample_id = str(value.get("id") or "").strip()
            if not sample_id:
                raise ValueError(f"result id is missing at {path}:{line_number}")
            existing = results.get(sample_id)
            if existing is None or _prefer_incoming_result(existing, value):
                results[sample_id] = value
    return results


def score_benchmark(
    samples: Sequence[BenchmarkSample],
    results: Mapping[str, Mapping[str, Any]],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    normalized_cutoffs = tuple(sorted({value for value in cutoffs if value > 0}))
    if not normalized_cutoffs:
        raise ValueError("at least one positive cutoff is required")
    sample_scores: list[dict[str, Any]] = []
    missing_result_ids: list[str] = []
    failed_result_ids: list[str] = []
    for sample in samples:
        result = results.get(sample.sample_id)
        if result is None:
            missing_result_ids.append(sample.sample_id)
            continue
        if result.get("error"):
            failed_result_ids.append(sample.sample_id)
            continue
        ranking = _result_ranking(result)
        sample_scores.append(score_sample(sample, ranking, cutoffs=normalized_cutoffs))

    known_sample_ids = {sample.sample_id for sample in samples}
    extra_result_ids = sorted(set(results) - known_sample_ids)
    by_task: dict[str, list[dict[str, Any]]] = {}
    by_split: dict[str, list[dict[str, Any]]] = {}
    for score in sample_scores:
        by_task.setdefault(str(score["task"]), []).append(score)
        by_split.setdefault(str(score["split"]), []).append(score)
    return {
        "sample_count": len(samples),
        "scored_count": len(sample_scores),
        "missing_result_ids": missing_result_ids,
        "failed_result_ids": failed_result_ids,
        "extra_result_ids": extra_result_ids,
        "cutoffs": list(normalized_cutoffs),
        "overall": _aggregate_scores(sample_scores, normalized_cutoffs),
        "by_task": {
            task: _aggregate_scores(scores, normalized_cutoffs)
            for task, scores in sorted(by_task.items())
        },
        "by_split": {
            split: _aggregate_scores(scores, normalized_cutoffs)
            for split, scores in sorted(by_split.items())
        },
        "samples": sample_scores,
    }


def score_sample(
    sample: BenchmarkSample,
    ranking: Sequence[str],
    *,
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    deduped_ranking = tuple(dict.fromkeys(ranking))
    relevance = {item.evidence_id: item.grade for item in sample.relevance}
    required = {item.evidence_id for item in sample.relevance if item.required}
    scores: dict[str, Any] = {
        "id": sample.sample_id,
        "split": sample.split,
        "task": sample.task,
        "annotation_status": sample.annotation.status,
        "retrieved_count": len(deduped_ranking),
    }
    for cutoff in cutoffs:
        retrieved = deduped_ranking[:cutoff]
        retrieved_set = set(retrieved)
        relevant_hits = retrieved_set & set(relevance)
        required_hits = retrieved_set & required
        scores[f"precision@{cutoff}"] = len(relevant_hits) / max(1, len(retrieved))
        scores[f"recall@{cutoff}"] = len(relevant_hits) / len(relevance)
        scores[f"required_recall@{cutoff}"] = (
            len(required_hits) / len(required) if required else 1.0
        )
        scores[f"mrr@{cutoff}"] = _reciprocal_rank(retrieved, relevance)
        scores[f"ndcg@{cutoff}"] = _ndcg(retrieved, relevance, cutoff)
        scores[f"coverage@{cutoff}"] = _coverage(
            retrieved_set,
            sample.coverage_groups,
        )
    return scores


def _aggregate_scores(
    scores: Sequence[Mapping[str, Any]],
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(scores)}
    for cutoff in cutoffs:
        for metric in ("precision", "recall", "required_recall", "mrr", "ndcg", "coverage"):
            key = f"{metric}@{cutoff}"
            result[key] = round(fmean(float(score[key]) for score in scores), 6) if scores else None
    return result


def _result_ranking(result: Mapping[str, Any]) -> list[str]:
    raw_evidence = result.get("evidence")
    if not isinstance(raw_evidence, list):
        return []
    ranking: list[str] = []
    for item in raw_evidence:
        if not isinstance(item, Mapping):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if evidence_id:
            ranking.append(evidence_id)
    return ranking


def _prefer_incoming_result(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> bool:
    existing_failed = bool(existing.get("error"))
    incoming_failed = bool(incoming.get("error"))
    if existing_failed != incoming_failed:
        return existing_failed and not incoming_failed
    return True


def _reciprocal_rank(ranking: Sequence[str], relevance: Mapping[str, int]) -> float:
    for rank, evidence_id in enumerate(ranking, 1):
        if evidence_id in relevance:
            return 1.0 / rank
    return 0.0


def _ndcg(ranking: Sequence[str], relevance: Mapping[str, int], cutoff: int) -> float:
    dcg = sum(
        (2 ** relevance.get(evidence_id, 0) - 1) / math.log2(rank + 1)
        for rank, evidence_id in enumerate(ranking[:cutoff], 1)
    )
    ideal_grades = sorted(relevance.values(), reverse=True)[:cutoff]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _coverage(retrieved: set[str], groups: Sequence[CoverageGroup]) -> float:
    if not groups:
        return 1.0
    satisfied = sum(len(retrieved & set(group.evidence_ids)) >= group.min_hits for group in groups)
    return satisfied / len(groups)


def _unique_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
