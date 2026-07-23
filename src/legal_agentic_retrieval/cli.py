from __future__ import annotations

import argparse
import json

from legal_agentic_retrieval.config import ModelConfig
from legal_agentic_retrieval.evidence import EvidencePacker
from legal_agentic_retrieval.graph import LegalRetrievalAgent
from legal_agentic_retrieval.index import CorpusIndexBuilder, RetrievalIndex
from legal_agentic_retrieval.models import Evidence, RetrievalRequest
from legal_agentic_retrieval.providers import (
    CohereReranker,
    OpenAIEmbedder,
    OpenAILegalPlanner,
)
from legal_agentic_retrieval.tokenization import TokenCounter


DEFAULT_INDEX = "data/corpus_v3.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone legal agentic retrieval")
    parser.add_argument("--env-file", default=".env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build compact metadata and vector index")
    build_parser.add_argument("--corpus-dir", default="corpus/structured")
    build_parser.add_argument("--index", default=DEFAULT_INDEX)

    query_parser = subparsers.add_parser("query", help="Run the LangGraph retrieval agent")
    query_parser.add_argument("question")
    query_parser.add_argument("--index", default=DEFAULT_INDEX)
    query_parser.add_argument("--top-k", type=int, default=10)
    query_parser.add_argument("--max-replans", type=int, default=1)
    query_parser.add_argument("--response-language", default="zh-CN")
    query_parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Return final agent-verified evidence without answer synthesis",
    )

    smoke_parser = subparsers.add_parser(
        "smoke", help="Verify configured LLM, embedding, and reranking endpoints"
    )
    smoke_parser.add_argument("--index", default=DEFAULT_INDEX)
    smoke_parser.add_argument("--question", default="精确检索 GDPR Article 6，并说明其适用法域")
    args = parser.parse_args()

    config = ModelConfig.from_env(args.env_file)
    embedder = OpenAIEmbedder(config)
    token_counter = TokenCounter(safety_factor=config.token_safety_factor)
    if args.command == "build":
        result = CorpusIndexBuilder(
            args.corpus_dir,
            embedder,
            token_counter=token_counter,
            passage_threshold_tokens=config.passage_threshold_tokens,
            passage_target_tokens=config.passage_target_tokens,
            passage_max_tokens=config.passage_max_tokens,
            passage_overlap_tokens=config.passage_overlap_tokens,
        ).build(args.index)
    elif args.command == "query":
        index = RetrievalIndex(args.index, embedder)
        agent = LegalRetrievalAgent(
            index,
            OpenAILegalPlanner(config),
            CohereReranker(config),
            evidence_packer=EvidencePacker(
                token_counter,
                total_budget=config.evidence_token_budget,
            ),
            max_replans=args.max_replans,
        )
        result = agent.invoke(
            RetrievalRequest(
                args.question,
                top_k=args.top_k,
                response_language=args.response_language,
                reference_only=args.reference_only,
            )
        )
    else:
        planner = OpenAILegalPlanner(config)
        reranker = CohereReranker(config)
        catalog = RetrievalIndex(args.index, embedder).catalog()["laws"]
        vector = embedder.embed(["legal retrieval endpoint smoke test"])
        plan = planner.plan(
            RetrievalRequest(args.question),
            corpus_catalog={
                "laws": catalog,
                "available_case_countries": [],
                "available_case_jurisdictions": [],
                "case_date_range": [None, None],
            },
        )
        sample_evidence = [
            Evidence(
                evidence_id="smoke:relevant",
                source_type="law_unit",
                title="Relevant sample",
                text="Lawful processing may be based on valid consent.",
                score=0.0,
                source_url=None,
                jurisdiction="EU",
                citation="Article 6",
            ),
            Evidence(
                evidence_id="smoke:irrelevant",
                source_type="case",
                title="Unrelated sample",
                text="A procedural scheduling note with no substantive privacy issue.",
                score=0.0,
                source_url=None,
            ),
        ]
        ranked = reranker.rerank(args.question, sample_evidence, top_n=2)
        result = {
            "llm": {"ok": True, "model": config.llm_model, "plan": plan.to_dict()},
            "embedding": {
                "ok": vector.shape == (1, config.embedding_dim),
                "model": config.embedding_model,
                "shape": list(vector.shape),
            },
            "reranker": {
                "ok": bool(ranked),
                "model": config.rerank_model,
                "ranking": [
                    {"evidence_id": item.evidence_id, "score": item.score} for item in ranked
                ],
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
