from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Callable, Literal, Mapping, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from legal_agentic_retrieval.models import RetrievalRequest
from legal_agentic_retrieval.providers import ProviderError


LOGGER = logging.getLogger(__name__)
SERVICE_NAME = "law-agentic-retrieval"


class AgentInvoker(Protocol):
    def invoke(self, request: RetrievalRequest) -> Mapping[str, Any]: ...


AgentFactory = Callable[[], AgentInvoker]


class QueryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=50_000)
    top_k: int = Field(default=10, ge=1, le=50)
    response_language: str = Field(default="zh-CN", min_length=1, max_length=64)
    reference_only: bool = False

    @field_validator("question", "response_language")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    index: str
    max_replans: int
    max_concurrency: int


class FindingResponse(BaseModel):
    title: str
    analysis: str
    risk_level: Any = None
    evidence_ids: list[str]
    uncertainty: Any = None


class AnswerResponse(BaseModel):
    summary: str
    findings: list[FindingResponse]
    limitations: list[str]
    disclaimer: str
    task: Literal["exact_law", "risk", "compare", "case_search"]
    plan: dict[str, Any]
    evidence_grade: dict[str, Any]
    evidence: list[dict[str, Any]]


class ReferenceOnlyResponse(BaseModel):
    reference_only: Literal[True]
    evidence: list[dict[str, Any]]


QueryResponse = AnswerResponse | ReferenceOnlyResponse


class _ServiceBusyError(RuntimeError):
    pass


class _Runtime:
    def __init__(self, agent: AgentInvoker, *, max_concurrency: int) -> None:
        self.agent = agent
        self._slots = BoundedSemaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="law-retrieval",
        )

    async def invoke(self, request: RetrievalRequest) -> Mapping[str, Any]:
        if not self._slots.acquire(blocking=False):
            raise _ServiceBusyError
        try:
            worker = self._executor.submit(self.agent.invoke, request)
        except Exception:
            self._slots.release()
            raise
        worker.add_done_callback(self._release_slot)
        # Synchronous model calls cannot be cancelled once dispatched. Polling the
        # concurrent future keeps the request coroutine cancellable while the slot
        # remains reserved until the underlying work actually finishes.
        while not worker.done():
            await asyncio.sleep(0.025)
        return worker.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _release_slot(self, _future: object) -> None:
        self._slots.release()


def create_app(
    agent_factory: AgentFactory,
    *,
    index_path: str,
    max_replans: int = 1,
    max_concurrency: int = 2,
) -> FastAPI:
    """Create an HTTP app that initializes one reusable retrieval agent at startup."""
    if max_replans < 0 or max_replans > 3:
        raise ValueError("max_replans must be between 0 and 3")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = _Runtime(
            agent_factory(),
            max_concurrency=max_concurrency,
        )
        app.state.runtime = runtime
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Law Centre Agent API",
        description="Evidence-grounded legal retrieval over the configured closed corpus.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            index=Path(index_path).name,
            max_replans=max_replans,
            max_concurrency=max_concurrency,
        )

    @app.post("/query", response_model=QueryResponse, tags=["retrieval"])
    async def query(payload: QueryPayload, request: Request) -> dict[str, Any]:
        retrieval_request = RetrievalRequest(
            text=payload.question,
            top_k=payload.top_k,
            response_language=payload.response_language,
            reference_only=payload.reference_only,
        )
        runtime: _Runtime = request.app.state.runtime
        try:
            return dict(await runtime.invoke(retrieval_request))
        except _ServiceBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "service_busy",
                    "message": "The legal retrieval service is at its concurrency limit.",
                },
                headers={"Retry-After": "1"},
            ) from exc
        except (OpenAIError, httpx.HTTPError, ProviderError) as exc:
            LOGGER.exception("A configured model or reranking dependency failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "code": "upstream_failure",
                    "message": "A configured model or reranking dependency failed.",
                },
            ) from exc
        except Exception as exc:
            LOGGER.exception("Unhandled legal retrieval failure")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "internal_error",
                    "message": "The legal retrieval request failed.",
                },
            ) from exc

    return app
