from __future__ import annotations

from pathlib import Path

from legal_agentic_retrieval.config import ModelConfig
from legal_agentic_retrieval.evidence import EvidencePacker
from legal_agentic_retrieval.graph import LegalRetrievalAgent
from legal_agentic_retrieval.index import RetrievalIndex
from legal_agentic_retrieval.providers import CohereReranker, OpenAIEmbedder, OpenAILegalPlanner
from legal_agentic_retrieval.tokenization import TokenCounter


def create_agent(
    config: ModelConfig,
    index_path: str | Path,
    *,
    max_replans: int = 1,
) -> LegalRetrievalAgent:
    """Create a query agent whose model clients and read-only index can be reused."""
    embedder = OpenAIEmbedder(config)
    token_counter = TokenCounter(safety_factor=config.token_safety_factor)
    index = RetrievalIndex(index_path, embedder)
    return LegalRetrievalAgent(
        index=index,
        planner=OpenAILegalPlanner(config),
        reranker=CohereReranker(config),
        evidence_packer=EvidencePacker(
            token_counter,
            total_budget=config.evidence_token_budget,
        ),
        max_replans=max_replans,
    )
