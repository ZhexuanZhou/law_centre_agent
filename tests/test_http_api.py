from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import sys
from threading import Event
from typing import Any

import httpx
import pytest
import uvicorn

from legal_agentic_retrieval import cli
from legal_agentic_retrieval.http_api import create_app
from legal_agentic_retrieval.models import RetrievalRequest
from legal_agentic_retrieval.providers import ProviderError


class RecordingAgent:
    def __init__(self) -> None:
        self.requests: list[RetrievalRequest] = []

    def invoke(self, request: RetrievalRequest) -> dict[str, Any]:
        self.requests.append(request)
        if request.reference_only:
            return {
                "reference_only": True,
                "evidence": [{"evidence_id": "law_unit:test"}],
            }
        return {
            "summary": "grounded",
            "findings": [],
            "limitations": [],
            "disclaimer": "review required",
            "task": "exact_law",
            "plan": {"task": "exact_law"},
            "evidence_grade": {"status": "sufficient"},
            "evidence": [],
        }


class FailingAgent:
    def invoke(self, request: RetrievalRequest) -> dict[str, Any]:
        raise ProviderError("upstream response contained secret diagnostic details")


class BrokenAgent:
    def invoke(self, request: RetrievalRequest) -> dict[str, Any]:
        raise ValueError("internal secret diagnostic details")


class BlockingAgent(RecordingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def invoke(self, request: RetrievalRequest) -> dict[str, Any]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the blocking agent")
        return super().invoke(request)


def _app(agent, **overrides):
    options = {
        "index_path": "/srv/law/index.sqlite3",
        "max_replans": 1,
        "max_concurrency": 4,
        **overrides,
    }
    return create_app(lambda: agent, **options)


@asynccontextmanager
async def _client(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


def test_health_reports_initialized_service_and_reuses_one_agent() -> None:
    agent = RecordingAgent()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return agent

    app = create_app(
        factory,
        index_path="/srv/law/index.sqlite3",
        max_replans=2,
        max_concurrency=3,
    )

    async def request_health():
        async with _client(app) as client:
            return await client.get("/health"), await client.get("/health")

    first, second = asyncio.run(request_health())

    assert first.status_code == 200
    assert first.json() == {
        "status": "ok",
        "service": "law-agentic-retrieval",
        "index": "index.sqlite3",
        "max_replans": 2,
        "max_concurrency": 3,
    }
    assert second.status_code == 200
    assert factory_calls == 1
    assert not agent.requests


def test_query_uses_defaults_and_returns_agent_result() -> None:
    agent = RecordingAgent()

    async def request_query():
        async with _client(_app(agent)) as client:
            return await client.post("/query", json={"question": "  GDPR 第六条  "})

    response = asyncio.run(request_query())

    assert response.status_code == 200
    assert response.json()["summary"] == "grounded"
    assert agent.requests == [
        RetrievalRequest(
            text="GDPR 第六条",
            top_k=10,
            response_language="zh-CN",
            reference_only=False,
        )
    ]


def test_query_forwards_all_options_and_reference_only_result() -> None:
    agent = RecordingAgent()

    async def request_query():
        async with _client(_app(agent)) as client:
            return await client.post(
                "/query",
                json={
                    "question": "Find the governing provision",
                    "top_k": 5,
                    "response_language": "en",
                    "reference_only": True,
                },
            )

    response = asyncio.run(request_query())

    assert response.status_code == 200
    assert response.json() == {
        "reference_only": True,
        "evidence": [{"evidence_id": "law_unit:test"}],
    }
    assert agent.requests[0].top_k == 5
    assert agent.requests[0].response_language == "en"
    assert agent.requests[0].reference_only is True


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "valid", "top_k": 0},
        {"question": "valid", "top_k": 51},
        {"question": "valid", "response_language": "  "},
        {"question": "x" * 50_001},
        {"question": "valid", "unknown": True},
    ],
)
def test_query_rejects_invalid_payloads(payload: dict[str, Any]) -> None:
    async def request_query():
        async with _client(_app(RecordingAgent())) as client:
            return await client.post("/query", json=payload)

    response = asyncio.run(request_query())

    assert response.status_code == 422


def test_query_hides_upstream_error_details() -> None:
    async def request_query():
        async with _client(_app(FailingAgent())) as client:
            return await client.post("/query", json={"question": "GDPR"})

    response = asyncio.run(request_query())

    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "upstream_failure",
            "message": "A configured model or reranking dependency failed.",
        }
    }
    assert "secret diagnostic" not in response.text


def test_query_maps_unexpected_errors_to_sanitized_500() -> None:
    async def request_query():
        async with _client(_app(BrokenAgent())) as client:
            return await client.post("/query", json={"question": "GDPR"})

    response = asyncio.run(request_query())

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "internal_error"
    assert "secret diagnostic" not in response.text


def test_query_rejects_requests_above_concurrency_limit() -> None:
    agent = BlockingAgent()

    async def request_queries():
        async with _client(_app(agent, max_concurrency=1)) as client:
            first_task = asyncio.create_task(client.post("/query", json={"question": "first"}))
            for _ in range(100):
                if agent.started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert agent.started.is_set()
            second = await client.post("/query", json={"question": "second"})
            agent.release.set()
            first = await first_task
            return first, second

    first, second = asyncio.run(request_queries())

    assert first.status_code == 200
    assert second.status_code == 503
    assert second.headers["retry-after"] == "1"
    assert second.json()["detail"]["code"] == "service_busy"


def test_cancelled_request_keeps_slot_until_agent_finishes() -> None:
    agent = BlockingAgent()

    async def request_queries():
        async with _client(_app(agent, max_concurrency=1)) as client:
            cancelled_task = asyncio.create_task(
                client.post("/query", json={"question": "cancelled"})
            )
            for _ in range(100):
                if agent.started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert agent.started.is_set()

            cancelled_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_task

            busy = await client.post("/query", json={"question": "still busy"})
            agent.release.set()
            for _ in range(100):
                if agent.requests:
                    break
                await asyncio.sleep(0.01)
            recovered = await client.post("/query", json={"question": "recovered"})
            return busy, recovered

    busy, recovered = asyncio.run(request_queries())

    assert busy.status_code == 503
    assert recovered.status_code == 200
    assert [request.text for request in agent.requests] == ["cancelled", "recovered"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_replans": 4}, "max_replans"),
        ({"max_concurrency": 0}, "max_concurrency"),
    ],
)
def test_create_app_rejects_invalid_service_limits(
    options: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _app(RecordingAgent(), **options)


def test_serve_parser_uses_safe_local_defaults() -> None:
    args = cli.build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.max_replans == 1
    assert args.max_concurrency == 2


def test_serve_command_runs_single_worker_uvicorn(monkeypatch, tmp_path) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    monkeypatch.setattr(
        cli.ModelConfig,
        "from_env",
        classmethod(lambda cls, env_file: object()),
    )
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "law-agentic-retrieval",
            "--env-file",
            str(tmp_path / ".env"),
            "serve",
            "--index",
            str(tmp_path / "index.sqlite3"),
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
            "--max-replans",
            "2",
            "--max-concurrency",
            "6",
            "--log-level",
            "warning",
        ],
    )

    cli.main()

    assert len(calls) == 1
    app, options = calls[0]
    assert app.title == "Law Centre Agent API"
    assert options == {
        "host": "0.0.0.0",
        "port": 9090,
        "log_level": "warning",
        "workers": 1,
    }


def test_openapi_documents_both_query_response_shapes() -> None:
    schema = _app(RecordingAgent()).openapi()
    response_schema = schema["paths"]["/query"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    variants = response_schema.get("anyOf") or response_schema.get("oneOf")
    assert variants is not None
    assert {item["$ref"].removeprefix("#/components/schemas/") for item in variants} == {
        "AnswerResponse",
        "ReferenceOnlyResponse",
    }
