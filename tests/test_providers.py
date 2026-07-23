from legal_agentic_retrieval.models import Evidence
from legal_agentic_retrieval.providers import CohereReranker, _validated_answer


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
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

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
