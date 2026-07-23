from collections import deque
from types import SimpleNamespace

import pytest

from legal_agentic_retrieval.models import Evidence
from legal_agentic_retrieval.models import RetrievalPlan
from legal_agentic_retrieval.providers import (
    CohereReranker,
    OpenAILegalPlanner,
    _parse_json_object,
    _validated_answer,
    _validated_grade,
)


def test_answer_validation_accepts_single_limitation_string_and_rejects_fake_evidence_id():
    evidence = Evidence(
        evidence_id="law_unit:1",
        source_type="law_unit",
        title="Law",
        text="Provision text",
        score=1.0,
        source_url=None,
    )

    result = _validated_answer(
        {
            "summary": "Summary",
            "findings": [
                {"title": "Supported", "analysis": "A", "evidence_ids": ["law_unit:1"]},
                {"title": "Invented", "analysis": "B", "evidence_ids": ["fake:1"]},
            ],
            "limitations": "Only a secondary source is available.",
        },
        [evidence],
    )

    assert result["limitations"] == ["Only a secondary source is available."]
    assert [item["title"] for item in result["findings"]] == ["Supported"]


def test_qwen_reranker_sends_instruction_separately_from_query(monkeypatch):
    captured = {}

    class Response:
        text = '{"results": [{"index": 0, "relevance_score": 0.9}]}'

        def raise_for_status(self):
            return None

    def fake_post(url, *, headers, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("legal_agentic_retrieval.providers.httpx.post", fake_post)
    reranker = CohereReranker.__new__(CohereReranker)
    reranker.url = "https://reranker.test/v1/rerank"
    reranker.model = "Qwen3-Reranker"
    reranker.api_key = "test"
    reranker.timeout = 1.0
    reranker.enabled = True
    reranker.min_score = 0.0
    reranker.instruction = "Retrieve directly relevant legal evidence."
    reranker.retries = 0
    evidence = [
        Evidence(
            evidence_id="case:1",
            source_type="case",
            title="Marketing case",
            text="The authority found unlawful direct marketing.",
            score=0.0,
            source_url=None,
        )
    ]

    ranked = reranker.rerank("find direct marketing cases", evidence, top_n=1)

    assert captured["instruction"] == "Retrieve directly relevant legal evidence."
    assert captured["query"] == "find direct marketing cases"
    assert ranked[0].evidence_id == "case:1"


def test_json_parser_recovers_fenced_object_with_trailing_commas() -> None:
    payload = _parse_json_object(
        """```json
        {"task": "case_search", "queries": ["cookie case",],}
        ```"""
    )

    assert payload == {"task": "case_search", "queries": ["cookie case"]}


def test_json_parser_does_not_promote_nested_object_from_truncated_response() -> None:
    with pytest.raises(ValueError, match="valid JSON object"):
        _parse_json_object(
            '{"task": "risk", "filters": {"source_types": ["law_unit", "case"]}, '
            '"exact_citations": [{"doc_id": "law"'
        )


def test_planner_retries_with_correction_prompt_after_invalid_json() -> None:
    contents = deque(
        [
            '{"queries": ["consent"]}',
            '{"task": "risk", "queries": ["consent"], "filters": {}}',
        ]
    )
    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=contents.popleft()),
                    )
                ]
            )

    planner = OpenAILegalPlanner.__new__(OpenAILegalPlanner)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    planner.model = "test"
    planner.temperature = 0.0
    planner.extra_body = {}
    planner.json_retries = 1

    payload = planner._json_completion(
        "Return JSON.",
        {"query": "risk"},
        max_tokens=100,
        validator=RetrievalPlan.from_mapping,
    )

    assert payload["task"] == "risk"
    assert len(calls) == 2
    assert "unsupported retrieval task" in calls[1]["messages"][-1]["content"]


def test_planner_accepts_schema_valid_object_nested_in_wrapper() -> None:
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"plan": {"task": "case_search", '
                                '"queries": ["cookie case"], "filters": {}}}'
                            )
                        ),
                    )
                ]
            )

    planner = OpenAILegalPlanner.__new__(OpenAILegalPlanner)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    planner.model = "test"
    planner.temperature = 0.0
    planner.extra_body = {}
    planner.json_retries = 0

    payload = planner._json_completion(
        "Return JSON.",
        {"query": "case"},
        max_tokens=100,
        validator=RetrievalPlan.from_mapping,
    )

    assert payload["task"] == "case_search"


def test_planner_retries_when_model_output_reaches_token_limit() -> None:
    responses = deque(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content='{"task": "risk", "queries": ["consent"]'),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=('{"task": "risk", "queries": ["consent"], "filters": {}}')
                        ),
                    )
                ]
            ),
        ]
    )
    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.popleft()

    planner = OpenAILegalPlanner.__new__(OpenAILegalPlanner)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    planner.model = "test"
    planner.temperature = 0.0
    planner.extra_body = {}
    planner.json_retries = 1

    payload = planner._json_completion(
        "Return JSON.",
        {"query": "risk"},
        max_tokens=100,
        validator=RetrievalPlan.from_mapping,
    )

    assert payload["task"] == "risk"
    assert len(calls) == 2
    assert "truncated at the output-token limit" in calls[1]["messages"][-1]["content"]


def test_reranker_retries_after_malformed_json(monkeypatch) -> None:
    responses = deque(
        [
            '{"results": [{"index": 0 "relevance_score": 0.9}]}',
            '{"results": [{"index": 0, "relevance_score": 0.9}]}',
        ]
    )

    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self):
            return None

    def fake_post(*args, **kwargs):
        return Response(responses.popleft())

    monkeypatch.setattr("legal_agentic_retrieval.providers.httpx.post", fake_post)
    reranker = CohereReranker.__new__(CohereReranker)
    reranker.url = "https://reranker.test/v1/rerank"
    reranker.model = "Qwen3-Reranker"
    reranker.api_key = "test"
    reranker.timeout = 1.0
    reranker.enabled = True
    reranker.min_score = 0.0
    reranker.instruction = "Rank legal evidence."
    reranker.retries = 1
    evidence = [
        Evidence(
            evidence_id="case:1",
            source_type="case",
            title="Case",
            text="Matching facts.",
            score=0.0,
            source_url=None,
        )
    ]

    ranked = reranker.rerank("find a case", evidence, top_n=1)

    assert ranked[0].evidence_id == "case:1"
    assert not responses


def test_reranker_retries_after_out_of_range_result_index(monkeypatch) -> None:
    responses = deque(
        [
            '{"results": [{"index": 9, "relevance_score": 0.9}]}',
            '{"results": [{"index": 0, "relevance_score": 0.8}]}',
        ]
    )

    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "legal_agentic_retrieval.providers.httpx.post",
        lambda *args, **kwargs: Response(responses.popleft()),
    )
    reranker = CohereReranker.__new__(CohereReranker)
    reranker.url = "https://reranker.test/v1/rerank"
    reranker.model = "Qwen3-Reranker"
    reranker.api_key = "test"
    reranker.timeout = 1.0
    reranker.enabled = True
    reranker.min_score = 0.0
    reranker.instruction = "Rank legal evidence."
    reranker.retries = 1
    evidence = [Evidence("case:1", "case", "Case", "Matching facts.", 0.0, None)]

    ranked = reranker.rerank("find a case", evidence, top_n=1)

    assert ranked[0].score == 0.8
    assert not responses


def test_risk_grade_requires_three_grounded_coverage_categories() -> None:
    plan = RetrievalPlan.from_mapping(
        {
            "task": "risk",
            "queries": ["assess consent risk"],
            "filters": {"source_types": ["law_unit", "case"]},
        }
    )
    evidence = [
        Evidence("law:primary", "law_unit", "Primary", "Rule", 1.0, None),
        Evidence("law:supplement", "law_unit", "Supplement", "Duty", 0.9, None),
        Evidence("case:analogous", "case", "Case", "Facts", 0.8, None),
    ]
    payload = {
        "status": "sufficient",
        "gaps": [],
        "requested_evidence_ids": [],
        "reasoning": "All categories are represented.",
        "coverage": {
            "applicable_law": {
                "satisfied": True,
                "evidence_ids": ["law:primary"],
                "gap": "",
            },
            "analogous_case": {
                "satisfied": True,
                "evidence_ids": ["case:analogous"],
                "gap": "",
            },
            "supplementary_obligations": {
                "satisfied": True,
                "evidence_ids": ["law:supplement"],
                "gap": "",
            },
        },
    }

    grade = _validated_grade(payload, plan, evidence)

    assert grade["sufficient"] is True
    assert all(item["satisfied"] for item in grade["coverage"].values())


def test_risk_grade_rejects_primary_law_reused_as_supplement() -> None:
    plan = RetrievalPlan.from_mapping(
        {
            "task": "risk",
            "queries": ["assess consent risk"],
            "filters": {"source_types": ["law_unit", "case"]},
        }
    )
    evidence = [
        Evidence("law:primary", "law_unit", "Primary", "Rule", 1.0, None),
        Evidence("case:analogous", "case", "Case", "Facts", 0.8, None),
    ]
    payload = {
        "status": "sufficient",
        "coverage": {
            "applicable_law": {
                "satisfied": True,
                "evidence_ids": ["law:primary"],
            },
            "analogous_case": {
                "satisfied": True,
                "evidence_ids": ["case:analogous"],
            },
            "supplementary_obligations": {
                "satisfied": True,
                "evidence_ids": ["law:primary"],
            },
        },
    }

    grade = _validated_grade(payload, plan, evidence)

    assert grade["sufficient"] is False
    assert grade["status"] == "retrieval_gap"
    assert any("supplementary_obligations" in gap for gap in grade["gaps"])
