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
        if plan.task == "risk":
            evidence = self._retrieve_risk_candidates(plan, request)
        else:
            evidence = self.index.exact(plan.exact_citations)
        if plan.task not in {"exact_law", "risk"} or (plan.task == "exact_law" and not evidence):
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
        dedupe = _dedupe_risk if plan.task == "risk" else _dedupe
        evidence = dedupe(evidence)
        case_ids = [
            str(item.metadata["case_id"]) for item in evidence if item.source_type == "case"
        ]
        if case_ids:
            evidence.extend(self.index.related_laws(case_ids, limit=request.top_k * 2))
        return {"evidence": [item.to_dict() for item in dedupe(evidence)]}

    def _retrieve_risk_candidates(
        self,
        plan: RetrievalPlan,
        request: RetrievalRequest,
    ) -> list[Evidence]:
        candidate_limit = _risk_candidate_limit(request.top_k)
        law_filters = replace(plan.filters, source_types=("law_unit",))
        case_filters = replace(
            plan.filters,
            jurisdictions=(),
            doc_ids=(),
            source_types=("case",),
        )
        applicable = [
            _with_retrieval_role(item, "applicable_law")
            for item in self.index.vector_search(
                plan.risk_queries.applicable_law,
                law_filters,
                limit=candidate_limit,
            )
        ]
        cases = [
            _with_retrieval_role(item, "analogous_case")
            for item in self.index.vector_search(
                plan.risk_queries.analogous_case,
                case_filters,
                limit=candidate_limit,
            )
        ]
        supplementary = [
            _with_retrieval_role(item, "supplementary_obligations")
            for item in self.index.vector_search(
                plan.risk_queries.supplementary_obligations,
                law_filters,
                limit=candidate_limit,
            )
        ]
        exact = [
            _with_risk_exact_rank(
                _with_retrieval_roles(
                    item,
                    ("applicable_law", "supplementary_obligations"),
                ),
                rank,
            )
            for rank, item in enumerate(self.index.exact(plan.exact_citations), 1)
        ]
        return _dedupe_risk([*applicable, *cases, *supplementary, *exact])

    def _rerank(self, state: AgentState) -> dict[str, Any]:
        request = RetrievalRequest(**state["request"])
        plan = RetrievalPlan.from_mapping(state["plan"])
        evidence = [Evidence(**item) for item in state.get("evidence", [])]
        if plan.task == "risk":
            ranked = self._rerank_risk(request, plan, evidence)
            return {"evidence": [item.to_dict() for item in ranked]}
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

    def _rerank_risk(
        self,
        request: RetrievalRequest,
        plan: RetrievalPlan,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        quotas = _risk_role_quotas(request.top_k)
        role_queries = {
            "applicable_law": plan.risk_queries.applicable_law,
            "analogous_case": plan.risk_queries.analogous_case,
            "supplementary_obligations": plan.risk_queries.supplementary_obligations,
        }
        rankings: dict[str, list[Evidence]] = {}
        for role, quota in quotas.items():
            if quota < 1:
                rankings[role] = []
                continue
            candidates = _risk_role_candidates(evidence, role)
            rerank_limit = min(len(candidates), max(quota * 3, 8))
            reranked = self.reranker.rerank(
                _risk_rerank_query(
                    request.text,
                    role,
                    role_queries[role],
                ),
                candidates,
                top_n=rerank_limit,
            )
            ranking = _reciprocal_rank_fusion(reranked, candidates)
            if role == "applicable_law":
                exact_intent = sorted(
                    (
                        item
                        for item in candidates
                        if isinstance(item.metadata.get("risk_exact_rank"), int)
                    ),
                    key=lambda item: int(item.metadata["risk_exact_rank"]),
                )
                ranking = _dedupe([*exact_intent[: min(quota, 2)], *ranking])
            rankings[role] = ranking
        return _merge_risk_rankings(
            rankings,
            evidence,
            quotas=quotas,
            limit=request.top_k,
        )

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
            if plan.task in {"exact_law", "compare"}
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


def _dedupe_risk(evidence: list[Evidence]) -> list[Evidence]:
    merged: dict[str, Evidence] = {}
    for item in evidence:
        current = merged.get(item.evidence_id)
        if current is None:
            merged[item.evidence_id] = item
            continue
        roles = tuple(
            dict.fromkeys(
                [
                    *_retrieval_roles(current),
                    *_retrieval_roles(item),
                ]
            )
        )
        preferred = item if item.score > current.score else current
        metadata = dict(preferred.metadata)
        if roles:
            metadata["retrieval_roles"] = list(roles)
        merged[item.evidence_id] = replace(
            preferred,
            metadata=metadata,
            matched_passage_ids=tuple(
                dict.fromkeys(
                    [
                        *current.matched_passage_ids,
                        *item.matched_passage_ids,
                    ]
                )
            ),
        )
    return list(merged.values())


def _with_retrieval_role(item: Evidence, role: str) -> Evidence:
    return _with_retrieval_roles(item, (role,))


def _with_retrieval_roles(item: Evidence, roles: tuple[str, ...]) -> Evidence:
    metadata = dict(item.metadata)
    metadata["retrieval_roles"] = list(dict.fromkeys([*_retrieval_roles(item), *roles]))
    return replace(item, metadata=metadata)


def _with_risk_exact_rank(item: Evidence, rank: int) -> Evidence:
    metadata = dict(item.metadata)
    metadata["risk_exact_rank"] = rank
    return replace(item, metadata=metadata)


def _retrieval_roles(item: Evidence) -> tuple[str, ...]:
    value = item.metadata.get("retrieval_roles")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(role) for role in value if str(role))


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


def _risk_candidate_limit(top_k: int) -> int:
    return max(top_k * 3, 12)


def _risk_role_quotas(limit: int) -> dict[str, int]:
    roles = (
        "applicable_law",
        "analogous_case",
        "supplementary_obligations",
    )
    quotas = {role: 0 for role in roles}
    for role in roles[:limit]:
        quotas[role] += 1
    distribution = (
        "applicable_law",
        "analogous_case",
        "applicable_law",
        "supplementary_obligations",
    )
    for index in range(max(0, limit - len(roles))):
        quotas[distribution[index % len(distribution)]] += 1
    return quotas


def _risk_role_candidates(evidence: list[Evidence], role: str) -> list[Evidence]:
    source_type = "case" if role == "analogous_case" else "law_unit"
    candidates: list[Evidence] = []
    for item in evidence:
        if item.source_type != source_type:
            continue
        roles = _retrieval_roles(item)
        if roles and role not in roles:
            continue
        candidates.append(item)
    return candidates


def _risk_rerank_query(
    user_query: str,
    role: str,
    queries: tuple[str, ...],
) -> str:
    objectives = {
        "applicable_law": (
            "Rank provisions that directly govern the main alleged risk. Prefer the rule whose "
            "elements match the conduct over adjacent recitals, enforcement powers, or remedies."
        ),
        "analogous_case": (
            "Rank case summaries by material factual similarity: jurisdiction, actor or industry, "
            "conduct, data use, procedural posture, and outcome. Prefer a close factual analogue "
            "over a case that merely cites the same legal provision."
        ),
        "supplementary_obligations": (
            "Rank additional duties independently raised by the facts and needed for a complete "
            "assessment. Exclude duplicates of the primary governing rule and generic provisions "
            "without a concrete connection to the scenario."
        ),
    }
    focused_queries = "\n".join(f"- {query}" for query in queries)
    return (
        f"Retrieval task: legal risk assessment.\nEvidence role: {role}.\n"
        f"Ranking objective: {objectives[role]}\n"
        f"Role-specific retrieval queries:\n{focused_queries}\n"
        f"Original user scenario: {user_query}"
    )


def _reciprocal_rank_fusion(
    reranked: list[Evidence],
    recalled: list[Evidence],
    *,
    rerank_weight: float = 0.25,
    rank_constant: int = 60,
) -> list[Evidence]:
    recalled_weight = 1.0 - rerank_weight
    evidence_by_id = {item.evidence_id: item for item in recalled}
    evidence_by_id.update(
        (item.evidence_id, evidence_by_id.get(item.evidence_id, item)) for item in reranked
    )
    reranked_positions = {item.evidence_id: rank for rank, item in enumerate(reranked, 1)}
    recalled_positions = {item.evidence_id: rank for rank, item in enumerate(recalled, 1)}
    reranked_fallback_rank = len(reranked) + 1
    recalled_fallback_rank = len(recalled) + 1
    scores = {
        evidence_id: (
            rerank_weight
            / (rank_constant + reranked_positions.get(evidence_id, reranked_fallback_rank))
            + recalled_weight
            / (rank_constant + recalled_positions.get(evidence_id, recalled_fallback_rank))
        )
        for evidence_id in evidence_by_id
    }
    return sorted(
        evidence_by_id.values(),
        key=lambda item: scores[item.evidence_id],
        reverse=True,
    )


def _merge_risk_rankings(
    rankings: Mapping[str, list[Evidence]],
    fallback: list[Evidence],
    *,
    quotas: Mapping[str, int],
    limit: int,
) -> list[Evidence]:
    role_order = (
        "applicable_law",
        "analogous_case",
        "supplementary_obligations",
    )
    selected: dict[str, list[Evidence]] = {role: [] for role in role_order}
    used: set[str] = set()
    for role in role_order:
        if quotas.get(role, 0) < 1:
            continue
        for item in rankings.get(role, []):
            if item.evidence_id in used:
                continue
            selected[role].append(item)
            used.add(item.evidence_id)
            if len(selected[role]) >= quotas.get(role, 0):
                break

    ordered: list[Evidence] = []
    for position in range(max((len(items) for items in selected.values()), default=0)):
        for role in role_order:
            if position < len(selected[role]):
                ordered.append(selected[role][position])

    fill = _dedupe(
        [
            *(item for role in role_order for item in rankings.get(role, [])),
            *fallback,
        ]
    )
    ordered.extend(item for item in fill if item.evidence_id not in used)
    return _dedupe(ordered)[:limit]


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
