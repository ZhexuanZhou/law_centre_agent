from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from legal_agentic_retrieval.evidence import EvidencePacker
from legal_agentic_retrieval.index import RetrievalIndex
from legal_agentic_retrieval.models import (
    AgentState,
    Evidence,
    RetrievalPlan,
    RetrievalRequest,
)
from legal_agentic_retrieval.providers import Planner, Reranker
from legal_agentic_retrieval.tokenization import TokenCounter


class LegalRetrievalAgent:
    """A bounded two-stage retrieval, context expansion, and synthesis workflow."""

    def __init__(
        self,
        index: RetrievalIndex,
        planner: Planner,
        reranker: Reranker,
        *,
        evidence_packer: EvidencePacker | None = None,
        max_replans: int = 1,
    ) -> None:
        if max_replans < 0 or max_replans > 3:
            raise ValueError("max_replans must be between 0 and 3")
        self.index = index
        self.planner = planner
        self.reranker = reranker
        self.evidence_packer = evidence_packer or EvidencePacker(TokenCounter())
        self.max_replans = max_replans
        self.graph = self._build()

    def invoke(self, request: RetrievalRequest | Mapping[str, Any]) -> dict[str, Any]:
        normalized = (
            request if isinstance(request, RetrievalRequest) else RetrievalRequest(**request)
        )
        state = self.graph.invoke({"request": asdict(normalized), "attempt": 0})
        retrieval_result = {
            "task": state["plan"]["task"],
            "plan": state["plan"],
            "evidence": state["evidence"],
        }
        if normalized.reference_only:
            return {"reference_only": True, "evidence": state["evidence"]}
        return {
            **state["answer"],
            **retrieval_result,
            "evidence_grade": state["grade"],
        }

    def _build(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("rerank", self._rerank)
        builder.add_node("hydrate", self._hydrate)
        builder.add_node("pack", self._pack)
        builder.add_node("grade", self._grade)
        builder.add_node("expand_context", self._expand_context)
        builder.add_node("replan", self._replan)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "hydrate")
        builder.add_edge("hydrate", "pack")
        builder.add_edge("pack", "grade")
        builder.add_conditional_edges(
            "grade",
            self._after_grade,
            {
                "expand_context": "expand_context",
                "replan": "replan",
                "references": END,
                "synthesize": "synthesize",
            },
        )
        builder.add_edge("expand_context", "grade")
        builder.add_edge("replan", "retrieve")
        builder.add_edge("synthesize", END)
        return builder.compile(name="law-agentic-retrieval")

    def _plan(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        plan = self.planner.plan(request, corpus_catalog=self.index.catalog())
        return {"plan": plan.to_dict()}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        plan = RetrievalPlan.from_mapping(state["plan"])
        evidence = self.index.exact(plan.exact_citations)
        if plan.task != "exact_law" or not evidence:
            semantic = self.index.vector_search(
                plan.queries,
                plan.filters,
                limit=max(request.top_k * 4, 20),
            )
            if (
                plan.task == "case_search"
                and not any(item.source_type == "case" for item in semantic)
                and plan.filters.doc_ids
            ):
                semantic = self.index.vector_search(
                    plan.queries,
                    replace(
                        plan.filters,
                        doc_ids=(),
                        source_types=("case",),
                    ),
                    limit=max(request.top_k * 4, 20),
                )
            evidence.extend(semantic)
        evidence = _dedupe(evidence)
        case_ids = [
            str(item.metadata["case_id"]) for item in evidence if item.source_type == "case"
        ]
        if case_ids:
            evidence.extend(self.index.related_laws(case_ids, limit=request.top_k * 2))
        return {"evidence": [item.to_dict() for item in _dedupe(evidence)]}

    def _rerank(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        plan = RetrievalPlan.from_mapping(state["plan"])
        evidence = [Evidence(**item) for item in state.get("evidence", [])]
        exact = self.index.exact(plan.exact_citations)
        exact_ids = {item.evidence_id for item in exact}
        semantic = [item for item in evidence if item.evidence_id not in exact_ids]
        rerank_limit = request.top_k
        if plan.task == "case_search":
            rerank_limit = min(len(semantic), max(request.top_k * 2, 10))
        reranked = self.reranker.rerank(
            _rerank_query(request.text, plan.task),
            semantic,
            top_n=rerank_limit,
        )
        if plan.task == "case_search":
            ranked = _merge_case_search(
                exact,
                reranked,
                semantic,
                reserve_law="law_unit" in plan.filters.source_types,
                limit=request.top_k,
            )
        else:
            ranked = _merge_with_source_coverage(
                exact,
                reranked,
                semantic,
                required_sources=_required_source_order(plan),
                limit=request.top_k,
            )
        return {"evidence": [item.to_dict() for item in ranked]}

    def _hydrate(self, state: AgentState) -> dict[str, Any]:
        evidence = [Evidence(**item) for item in state.get("evidence", [])]
        hydrated = self.index.hydrate(evidence)
        return {"hydrated_evidence": [item.to_dict() for item in hydrated]}

    def _pack(self, state: AgentState) -> dict[str, Any]:
        return self._pack_evidence(state)

    def _grade(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        plan = RetrievalPlan.from_mapping(state["plan"])
        evidence = [Evidence(**item) for item in state.get("evidence", [])]
        grade = self.planner.grade(request, plan, evidence)
        return {"grade": grade}

    def _after_grade(self, state: AgentState) -> str:
        grade = state.get("grade", {})
        sufficient = bool(grade.get("sufficient"))
        if sufficient or state.get("attempt", 0) >= self.max_replans:
            request = RetrievalRequest(**state["request"])
            return "references" if request.reference_only else "synthesize"
        if grade.get("status") == "context_gap":
            return "expand_context"
        return "replan"

    def _expand_context(self, state: AgentState) -> dict[str, Any]:
        result = self._pack_evidence(
            state,
            priority_ids=state.get("grade", {}).get("requested_evidence_ids") or [],
        )
        result["attempt"] = state.get("attempt", 0) + 1
        return result

    def _replan(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        previous = RetrievalPlan.from_mapping(state["plan"])
        plan = self.planner.plan(
            request,
            corpus_catalog=self.index.catalog(),
            previous_plan=previous,
            gaps=state.get("grade", {}).get("gaps") or [],
        )
        return {"plan": plan.to_dict(), "attempt": state.get("attempt", 0) + 1}

    def _pack_evidence(
        self,
        state: AgentState,
        *,
        priority_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        plan = RetrievalPlan.from_mapping(state["plan"])
        hydrated = [Evidence(**item) for item in state.get("hydrated_evidence", [])]
        exact_ids = (
            {item.evidence_id for item in self.index.exact(plan.exact_citations)}
            if plan.task != "case_search"
            else set()
        )
        packed = self.evidence_packer.pack(
            hydrated,
            exact_ids=exact_ids,
            priority_ids=priority_ids or [],
        )
        return {"evidence": [item.to_dict() for item in packed]}

    def _synthesize(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        plan = RetrievalPlan.from_mapping(state["plan"])
        evidence = [Evidence(**item) for item in state.get("evidence", [])]
        answer = self.planner.synthesize(
            request,
            plan,
            evidence,
            state.get("grade", {}),
        )
        return {"answer": answer}


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    return list({item.evidence_id: item for item in evidence}.values())


def _rerank_query(query: str, task: str) -> str:
    if task == "case_search":
        return (
            "Retrieval task: case search. Rank factually matching case summaries before "
            f"supporting legal provisions.\nUser query: {query}"
        )
    if task == "risk":
        return (
            "Retrieval task: legal risk assessment. Relevant evidence may cover the directly "
            "applicable law, analogous cases, and supplementary obligations.\n"
            f"User query: {query}"
        )
    return query


def _required_source_order(plan: RetrievalPlan) -> tuple[str, ...]:
    sources = plan.filters.source_types
    if plan.task != "case_search" or "case" not in sources:
        return sources
    return ("case", *(source for source in sources if source != "case"))


def _merge_case_search(
    exact: list[Evidence],
    reranked: list[Evidence],
    candidates: list[Evidence],
    *,
    reserve_law: bool,
    limit: int,
) -> list[Evidence]:
    ordered = _dedupe([*reranked, *candidates, *exact])
    cases = [item for item in ordered if item.source_type == "case"]
    laws = _dedupe(
        [
            *exact,
            *(item for item in ordered if item.source_type == "law_unit"),
        ]
    )
    case_limit = limit
    if reserve_law and laws and limit > 1:
        case_limit -= 1
    selected_cases = cases[:case_limit]
    return _dedupe([*selected_cases, *laws, *cases])[:limit]


def _merge_with_source_coverage(
    exact: list[Evidence],
    reranked: list[Evidence],
    candidates: list[Evidence],
    *,
    required_sources: tuple[str, ...],
    limit: int,
) -> list[Evidence]:
    coverage: list[Evidence] = []
    represented = {item.source_type for item in exact}
    for source_type in required_sources:
        if source_type in represented:
            continue
        source_candidates = [item for item in reranked if item.source_type == source_type]
        if not source_candidates:
            source_candidates = sorted(
                (item for item in candidates if item.source_type == source_type),
                key=lambda item: item.score,
                reverse=True,
            )
        if source_candidates:
            coverage.append(source_candidates[0])
            represented.add(source_type)
    return _dedupe([*coverage, *exact, *reranked, *candidates])[:limit]
