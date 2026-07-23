"""Standalone, model-driven agentic retrieval for a structured legal corpus."""

from legal_agentic_retrieval.graph import LegalRetrievalAgent
from legal_agentic_retrieval.index import CorpusIndexBuilder, RetrievalIndex
from legal_agentic_retrieval.models import RetrievalRequest

__all__ = [
    "CorpusIndexBuilder",
    "LegalRetrievalAgent",
    "RetrievalIndex",
    "RetrievalRequest",
]
