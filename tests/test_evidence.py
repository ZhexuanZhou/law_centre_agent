from legal_agentic_retrieval.evidence import EvidencePacker
from legal_agentic_retrieval.models import Evidence
from legal_agentic_retrieval.tokenization import TokenCounter


def test_packer_preserves_every_candidate_under_shared_budget():
    evidence = [
        Evidence(
            evidence_id=f"law_unit:{index}",
            source_type="law_unit",
            title=f"Law {index}",
            text="legal requirement " * 400,
            score=float(10 - index),
            source_url=None,
            jurisdiction=str(index),
        )
        for index in range(4)
    ]
    packer = EvidencePacker(
        TokenCounter(safety_factor=1.0),
        total_budget=400,
        law_limit=400,
        min_record_budget=50,
    )

    packed = packer.pack(evidence)

    assert [item.evidence_id for item in packed] == [
        "law_unit:0",
        "law_unit:1",
        "law_unit:2",
        "law_unit:3",
    ]
    assert all(item.included_tokens > 0 for item in packed)
    assert sum(item.included_tokens for item in packed) <= 400


def test_packer_preserves_reranked_order_instead_of_resorting_mixed_scores():
    evidence = [
        Evidence(
            evidence_id="case:relevant",
            source_type="case",
            title="Relevant case",
            text="Factually analogous decision.",
            score=0.6,
            source_url=None,
        ),
        Evidence(
            evidence_id="law_unit:related",
            source_type="law_unit",
            title="Related law",
            text="Supporting provision.",
            score=1.0,
            source_url=None,
        ),
    ]

    packed = EvidencePacker(TokenCounter(safety_factor=1.0)).pack(evidence)

    assert [item.evidence_id for item in packed] == [
        "case:relevant",
        "law_unit:related",
    ]
