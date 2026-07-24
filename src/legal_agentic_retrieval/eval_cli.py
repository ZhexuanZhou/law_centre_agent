from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from legal_agentic_retrieval.config import ModelConfig
from legal_agentic_retrieval.evaluation import (
    BenchmarkSample,
    export_annotation_csv,
    load_benchmark,
    load_results,
    score_benchmark,
    validate_benchmark,
)
from legal_agentic_retrieval.evidence import EvidencePacker
from legal_agentic_retrieval.graph import LegalRetrievalAgent
from legal_agentic_retrieval.index import RetrievalIndex
from legal_agentic_retrieval.models import RetrievalRequest
from legal_agentic_retrieval.providers import CohereReranker, OpenAIEmbedder, OpenAILegalPlanner
from legal_agentic_retrieval.tokenization import TokenCounter


DEFAULT_DATASET = "evals/benchmark_v0.jsonl"
DEFAULT_INDEX = "data/corpus_v3.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate, run, and score legal retrieval evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    validate_parser.add_argument("--index", default=DEFAULT_INDEX)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    run_parser.add_argument("--index", default=DEFAULT_INDEX)
    run_parser.add_argument("--env-file", default=".env")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--task", choices=["exact_law", "risk", "compare", "case_search"])
    run_parser.add_argument("--split", choices=["dev", "test"])
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--top-k", type=int)
    run_parser.add_argument("--max-replans", type=int, default=1)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--with-answer", action="store_true")

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    score_parser.add_argument("--results", required=True)
    score_parser.add_argument("--cutoff", action="append", type=int)
    score_parser.add_argument(
        "--task",
        choices=["exact_law", "risk", "compare", "case_search"],
    )
    score_parser.add_argument("--split", choices=["dev", "test"])

    export_parser = subparsers.add_parser(
        "export-csv",
        help="Export query-evidence rows for an external annotation service",
    )
    export_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    export_parser.add_argument("--index", default=DEFAULT_INDEX)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--task", choices=["exact_law", "risk", "compare", "case_search"])
    export_parser.add_argument("--split", choices=["dev", "test"])
    export_parser.add_argument(
        "--text-limit",
        type=int,
        default=0,
        help="Maximum characters in the combined evidence cell; 0 keeps the complete text",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        result = validate_benchmark(load_benchmark(args.dataset), index_path=args.index)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
        return
    if args.command == "score":
        samples = _select_samples(
            load_benchmark(args.dataset),
            task=args.task,
            split=args.split,
        )
        result = score_benchmark(
            samples,
            load_results(args.results),
            cutoffs=args.cutoff or (1, 3, 5, 10),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "export-csv":
        samples = _select_samples(
            load_benchmark(args.dataset),
            task=args.task,
            split=args.split,
        )
        result = export_annotation_csv(
            samples,
            index_path=args.index,
            output_path=args.output,
            text_limit=args.text_limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    _run(args)


def _run(args: argparse.Namespace) -> None:
    samples = _select_samples(
        load_benchmark(args.dataset),
        task=args.task,
        split=args.split,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("no benchmark samples selected")

    output = Path(args.output)
    completed_ids = set()
    if args.resume and output.is_file():
        completed_ids = {
            sample_id
            for sample_id, result in load_results(output).items()
            if not result.get("error")
        }
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    config = ModelConfig.from_env(args.env_file)
    embedder = OpenAIEmbedder(config)
    token_counter = TokenCounter(safety_factor=config.token_safety_factor)
    agent = LegalRetrievalAgent(
        RetrievalIndex(args.index, embedder),
        OpenAILegalPlanner(config),
        CohereReranker(config),
        evidence_packer=EvidencePacker(
            token_counter,
            total_budget=config.evidence_token_budget,
        ),
        max_replans=args.max_replans,
    )

    failures = 0
    for position, sample in enumerate(samples, 1):
        if sample.sample_id in completed_ids:
            continue
        started_at = time.perf_counter()
        try:
            result: dict[str, Any] = agent.invoke(
                RetrievalRequest(
                    sample.query,
                    top_k=args.top_k or sample.retrieval_k,
                    response_language=sample.language,
                    reference_only=not args.with_answer,
                )
            )
            record = {
                "id": sample.sample_id,
                "query": sample.query,
                "split": sample.split,
                "expected_task": sample.task,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                **result,
            }
        except Exception as exc:
            failures += 1
            record = {
                "id": sample.sample_id,
                "query": sample.query,
                "split": sample.split,
                "expected_task": sample.task,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[{position}/{len(samples)}] {sample.sample_id} "
            f"{'failed' if record.get('error') else 'completed'}"
        )
    if failures:
        raise SystemExit(1)


def _select_samples(
    samples: list[BenchmarkSample],
    *,
    task: str | None,
    split: str | None,
) -> list[BenchmarkSample]:
    selected = samples
    if task:
        selected = [sample for sample in selected if sample.task == task]
    if split:
        selected = [sample for sample in selected if sample.split == split]
    return selected


if __name__ == "__main__":
    main()
