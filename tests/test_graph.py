from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

from legal_agentic_retrieval.evidence import EvidencePacker
from legal_agentic_retrieval.graph import (
    LegalRetrievalAgent,
    _merge_risk_rankings,
    _reciprocal_rank_fusion,
    _risk_role_quotas,
)
from legal_agentic_retrieval.index import RetrievalIndex
from legal_agentic_retrieval.models import Evidence, RetrievalPlan, RetrievalRequest
from legal_agentic_retrieval.tokenization import TokenCounter


class FakePlanner:
    def __init__(
        self,
        plans: Sequence[RetrievalPlan],
        grades: Sequence[bool | Mapping[str, Any]],
    ) -> None:
        self.plans = deque(plans)
        self.grades = deque(grades)
        self.plan_calls = 0
        self.grade_evidence: list[list[Evidence]] = []
        self.synthesize_calls = 0

    def plan(
        self,
        request: RetrievalRequest,
        *,
        corpus_catalog: Mapping[str, Any],
        previous_plan: RetrievalPlan | None = None,
        gaps: Sequence[str] = (),
    ) -> RetrievalPlan:
        self.plan_calls += 1
        assert corpus_catalog["laws"]
        return self.plans.popleft()

    def grade(
        self,
        request: RetrievalRequest,
        plan: RetrievalPlan,
        evidence: Sequence[Evidence],
    ) -> dict[str, Any]:
        self.grade_evidence.append(list(evidence))
        grade = self.grades.popleft()
        if isinstance(grade, Mapping):
            return dict(grade)
        return {
            "status": "sufficient" if grade else "retrieval_gap",
            "sufficient": grade,
            "gaps": [] if grade else ["missing scope"],
        }

    def synthesize(
        self,
        request: RetrievalRequest,
        plan: RetrievalPlan,
        evidence: Sequence[Evidence],
        grade: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert request.response_language == "zh-CN"
        self.synthesize_calls += 1
        return {
            "summary": "grounded",
            "findings": [
                {
                    "title": "result",
                    "analysis": "based on evidence",
                    "risk_level": None,
                    "evidence_ids": [evidence[0].evidence_id],
                    "uncertainty": None,
                }
            ]
            if evidence
            else [],
            "limitations": [],
            "disclaimer": "review required",
        }


class ScoreReranker:
    def rerank(self, query: str, evidence: Sequence[Evidence], *, top_n: int) -> list[Evidence]:
        return list(evidence[:top_n])


class LawOnlyReranker:
    def rerank(self, query: str, evidence: Sequence[Evidence], *, top_n: int) -> list[Evidence]:
        return [item for item in evidence if item.source_type == "law_unit"][:top_n]


class RecordingRoleReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(
        self,
        query: str,
        evidence: Sequence[Evidence],
        *,
        top_n: int,
    ) -> list[Evidence]:
        self.queries.append(query)
        return list(evidence[:top_n])


def _plan(**overrides: Any) -> RetrievalPlan:
    payload = {
        "task": "risk",
        "queries": ["marketing consent"],
        "filters": {"source_types": ["law_unit", "case"]},
        "exact_citations": [],
        "comparison_targets": [],
        **overrides,
    }
    return RetrievalPlan.from_mapping(payload)


def test_graph_retrieves_cases_and_expands_resolved_law_relation(built_index):
    path, embedder = built_index
    planner = FakePlanner([_plan()], [True])
    agent = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, ScoreReranker(), max_replans=1
    )

    result = agent.invoke(RetrievalRequest("Find marketing consent risks", top_k=5))

    assert result["task"] == "risk"
    assert {item["source_type"] for item in result["evidence"]} == {"law_unit", "case"}
    assert result["findings"][0]["evidence_ids"][0] in {
        item["evidence_id"] for item in result["evidence"]
    }


def test_risk_retrieval_uses_role_queries_candidate_quotas_and_interleaved_slots(
    built_index,
    monkeypatch,
):
    path, embedder = built_index
    planner = FakePlanner(
        [
            _plan(
                filters={
                    "doc_ids": ["eu_gdpr_2016_679", "china_pipl_2021"],
                    "source_types": ["law_unit", "case"],
                },
                risk_queries={
                    "applicable_law": ["direct consent rule"],
                    "analogous_case": ["retail marketing without consent"],
                    "supplementary_obligations": ["proof and accountability duty"],
                },
            )
        ],
        [True],
    )
    reranker = RecordingRoleReranker()
    index = RetrievalIndex(path, embedder)
    original_vector_search = index.vector_search
    searches: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]] = []

    def recording_vector_search(queries, filters, *, limit):
        searches.append(
            (
                tuple(queries),
                tuple(filters.source_types),
                tuple(filters.doc_ids),
                limit,
            )
        )
        return original_vector_search(queries, filters, limit=limit)

    monkeypatch.setattr(index, "vector_search", recording_vector_search)
    result = LegalRetrievalAgent(
        index,
        planner,
        reranker,
        max_replans=0,
    ).invoke(RetrievalRequest("Assess consent risk", top_k=3))

    assert searches == [
        (
            ("direct consent rule",),
            ("law_unit",),
            ("eu_gdpr_2016_679", "china_pipl_2021"),
            12,
        ),
        (("retail marketing without consent",), ("case",), (), 12),
        (
            ("proof and accountability duty",),
            ("law_unit",),
            ("eu_gdpr_2016_679", "china_pipl_2021"),
            12,
        ),
    ]
    assert [item["source_type"] for item in result["evidence"]] == [
        "law_unit",
        "case",
        "law_unit",
    ]
    assert len(reranker.queries) == 3
    assert any("Evidence role: applicable_law" in query for query in reranker.queries)
    assert any("Evidence role: analogous_case" in query for query in reranker.queries)
    assert any("Evidence role: supplementary_obligations" in query for query in reranker.queries)


def test_risk_slot_quota_and_merge_keep_roles_distinct():
    quotas = _risk_role_quotas(8)
    laws = [
        Evidence(f"law:{index}", "law_unit", f"Law {index}", "Rule", 1.0, None)
        for index in range(6)
    ]
    cases = [
        Evidence(f"case:{index}", "case", f"Case {index}", "Facts", 1.0, None) for index in range(3)
    ]

    merged = _merge_risk_rankings(
        {
            "applicable_law": laws[:4],
            "analogous_case": cases,
            "supplementary_obligations": [laws[0], laws[4], laws[5]],
        },
        [*laws, *cases],
        quotas=quotas,
        limit=8,
    )

    assert quotas == {
        "applicable_law": 4,
        "analogous_case": 2,
        "supplementary_obligations": 2,
    }
    assert [item.source_type for item in merged] == [
        "law_unit",
        "case",
        "law_unit",
        "law_unit",
        "case",
        "law_unit",
        "law_unit",
        "law_unit",
    ]
    assert len({item.evidence_id for item in merged}) == 8


def test_risk_rank_fusion_prevents_reranker_from_dropping_top_recall_candidate():
    recalled = [
        Evidence("case:gold", "case", "Gold", "Closest facts", 0.9, None),
        Evidence("case:b", "case", "B", "Related", 0.8, None),
        Evidence("case:c", "case", "C", "Related", 0.7, None),
        Evidence("case:d", "case", "D", "Related", 0.6, None),
    ]
    reranked = [recalled[1], recalled[2], recalled[3], recalled[0]]

    fused = _reciprocal_rank_fusion(reranked, recalled)

    assert fused.index(recalled[0]) < 3


def test_risk_rerank_promotes_only_primary_exact_intent():
    generic = Evidence(
        "law:generic",
        "law_unit",
        "Generic",
        "Adjacent rule",
        0.9,
        None,
        metadata={"retrieval_roles": ["applicable_law"]},
    )
    primary = Evidence(
        "law:primary",
        "law_unit",
        "Primary",
        "Direct rule",
        1.0,
        None,
        metadata={
            "retrieval_roles": ["applicable_law", "supplementary_obligations"],
            "risk_exact_rank": 1,
        },
    )
    supplementary = Evidence(
        "law:supplement",
        "law_unit",
        "Supplement",
        "Additional duty",
        0.8,
        None,
        metadata={"retrieval_roles": ["supplementary_obligations"]},
    )
    analogous = Evidence(
        "case:analogous",
        "case",
        "Analogous",
        "Matching facts",
        0.8,
        None,
        metadata={"retrieval_roles": ["analogous_case"]},
    )
    agent = LegalRetrievalAgent.__new__(LegalRetrievalAgent)
    agent.reranker = RecordingRoleReranker()

    ranked = agent._rerank_risk(
        RetrievalRequest("Assess risk", top_k=3),
        _plan(
            risk_queries={
                "applicable_law": ["direct"],
                "analogous_case": ["facts"],
                "supplementary_obligations": ["additional"],
            }
        ),
        [generic, primary, supplementary, analogous],
    )

    assert [item.evidence_id for item in ranked] == [
        "law:primary",
        "case:analogous",
        "law:supplement",
    ]


def test_graph_replans_once_when_model_grades_evidence_insufficient(built_index):
    path, embedder = built_index
    planner = FakePlanner(
        [
            _plan(filters={"source_types": ["case"], "countries": ["Nowhere"]}),
            _plan(filters={"source_types": ["law_unit", "case"]}),
        ],
        [False, True],
    )
    agent = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, ScoreReranker(), max_replans=1
    )

    result = agent.invoke(RetrievalRequest("Assess risk", top_k=5))

    assert planner.plan_calls == 2
    assert result["evidence_grade"]["sufficient"] is True
    assert result["evidence"]


def test_graph_preserves_exact_evidence_when_reranker_returns_nothing(built_index):
    path, embedder = built_index
    planner = FakePlanner(
        [
            _plan(
                task="exact_law",
                queries=["Article 6"],
                exact_citations=[{"doc_id": "eu_gdpr_2016_679", "local_citation": "Article 6"}],
            )
        ],
        [True],
    )

    result = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, ScoreReranker(), max_replans=0
    ).invoke(RetrievalRequest("Retrieve Article 6", top_k=5))

    assert [item["evidence_id"] for item in result["evidence"]] == ["law_unit:gdpr:article_6"]


def test_graph_preserves_requested_source_coverage_after_reranking(built_index):
    path, embedder = built_index
    planner = FakePlanner([_plan()], [True])

    result = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, LawOnlyReranker(), max_replans=0
    ).invoke(RetrievalRequest("Assess marketing consent", top_k=5))

    assert {item["source_type"] for item in result["evidence"]} == {"law_unit", "case"}


def test_case_search_reserves_case_slot_when_exact_laws_fill_top_k(built_index):
    path, embedder = built_index
    planner = FakePlanner(
        [
            _plan(
                task="case_search",
                filters={"source_types": ["case"]},
                exact_citations=[
                    {"doc_id": "eu_gdpr_2016_679", "local_citation": "Article 6"},
                    {"doc_id": "china_pipl_2021", "local_citation": "Article 13"},
                ],
            )
        ],
        [True],
    )

    result = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, ScoreReranker(), max_replans=0
    ).invoke(RetrievalRequest("Find marketing cases", top_k=2))

    assert len(result["evidence"]) == 2
    assert result["evidence"][0]["source_type"] == "case"
    assert "case" in {item["source_type"] for item in result["evidence"]}


def test_case_search_retries_without_indirect_doc_id_filter(built_index):
    path, embedder = built_index
    planner = FakePlanner(
        [
            _plan(
                task="case_search",
                filters={
                    "source_types": ["case"],
                    "doc_ids": ["china_pipl_2021"],
                },
            )
        ],
        [True],
    )

    result = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, ScoreReranker(), max_replans=0
    ).invoke(RetrievalRequest("Find marketing cases", top_k=2))

    assert "case:gdprhub:1" in {item["evidence_id"] for item in result["evidence"]}


def test_case_search_plan_cannot_drop_case_source():
    plan = _plan(
        task="case_search",
        filters={"source_types": ["law_unit"]},
    )

    assert "case" in plan.filters.source_types


def test_reference_only_stops_after_final_evidence_pack(built_index):
    path, embedder = built_index
    planner = FakePlanner(
        [
            _plan(
                task="case_search",
                filters={"source_types": ["case"]},
            )
        ],
        [True],
    )
    agent = LegalRetrievalAgent(
        RetrievalIndex(path, embedder), planner, ScoreReranker(), max_replans=1
    )

    result = agent.invoke(
        RetrievalRequest(
            "Find marketing cases",
            top_k=2,
            reference_only=True,
        )
    )

    assert set(result) == {"reference_only", "evidence"}
    assert result["reference_only"] is True
    assert result["evidence"]
    assert planner.grade_evidence
    assert planner.synthesize_calls == 0


def test_graph_expands_requested_context_without_replanning(built_index):
    path, embedder = built_index
    requested_id = "law_unit:gdpr:article_6"
    planner = FakePlanner(
        [_plan()],
        [
            {
                "status": "context_gap",
                "sufficient": False,
                "gaps": ["the provision is truncated by context budget"],
                "requested_evidence_ids": [requested_id],
            },
            {
                "status": "sufficient",
                "sufficient": True,
                "gaps": [],
                "requested_evidence_ids": [],
            },
        ],
    )
    packer = EvidencePacker(
        TokenCounter(safety_factor=1.0),
        total_budget=200,
        law_limit=20,
        case_limit=20,
        expanded_limit=100,
        min_record_budget=30,
    )
    agent = LegalRetrievalAgent(
        RetrievalIndex(path, embedder),
        planner,
        ScoreReranker(),
        evidence_packer=packer,
        max_replans=1,
    )

    result = agent.invoke(RetrievalRequest("Assess marketing consent", top_k=5))

    first = next(item for item in planner.grade_evidence[0] if item.evidence_id == requested_id)
    second = next(item for item in planner.grade_evidence[1] if item.evidence_id == requested_id)
    assert planner.plan_calls == 1
    assert first.is_truncated is True
    assert second.included_tokens > first.included_tokens
    assert result["evidence_grade"]["status"] == "sufficient"
