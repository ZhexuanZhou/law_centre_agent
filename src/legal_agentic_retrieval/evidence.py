from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Sequence

from legal_agentic_retrieval.models import Evidence
from legal_agentic_retrieval.tokenization import TokenCounter


class EvidencePacker:
    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        total_budget: int = 42_000,
        exact_law_limit: int = 12_000,
        law_limit: int = 6_000,
        case_limit: int = 8_000,
        expanded_limit: int = 16_000,
        min_record_budget: int = 256,
    ) -> None:
        if total_budget < 1:
            raise ValueError("total_budget must be positive")
        if min_record_budget < 1:
            raise ValueError("min_record_budget must be positive")
        self.counter = token_counter
        self.total_budget = total_budget
        self.exact_law_limit = exact_law_limit
        self.law_limit = law_limit
        self.case_limit = case_limit
        self.expanded_limit = expanded_limit
        self.min_record_budget = min_record_budget

    def pack(
        self,
        evidence: Sequence[Evidence],
        *,
        exact_ids: Iterable[str] = (),
        priority_ids: Iterable[str] = (),
    ) -> list[Evidence]:
        exact = set(exact_ids)
        priority = set(priority_ids)
        ordered = [
            item
            for _, item in sorted(
                enumerate(evidence),
                key=lambda indexed: (
                    indexed[1].evidence_id not in priority,
                    indexed[1].evidence_id not in exact,
                    indexed[0],
                ),
            )
        ]
        packed: list[Evidence] = []
        remaining = self.total_budget
        for index, item in enumerate(ordered):
            if remaining < 1:
                break
            per_item = self._item_limit(item, exact=exact, priority=priority)
            records_after = len(ordered) - index - 1
            reserve = min(self.min_record_budget * records_after, remaining - 1)
            allowed = min(per_item, max(remaining - reserve, 1))
            packed_item = self._pack_item(item, allowed)
            if packed_item.included_tokens < 1:
                continue
            packed.append(packed_item)
            remaining -= packed_item.included_tokens
        return packed

    def _item_limit(self, item: Evidence, *, exact: set[str], priority: set[str]) -> int:
        if item.evidence_id in priority:
            return self.expanded_limit
        if item.evidence_id in exact and item.source_type == "law_unit":
            return self.exact_law_limit
        return self.law_limit if item.source_type == "law_unit" else self.case_limit

    def _pack_item(self, item: Evidence, allowed: int) -> Evidence:
        original_tokens = self.counter.count(item.text)
        clean_metadata = dict(item.metadata)
        passages = clean_metadata.pop("passages", [])
        if original_tokens <= allowed:
            return replace(
                item,
                content_mode="full",
                original_tokens=original_tokens,
                included_tokens=original_tokens,
                is_truncated=False,
                omission_reason=None,
                metadata=clean_metadata,
            )

        selected_text, selected_ids = self._select_passages(item, passages, allowed)
        if not selected_text:
            selected_text = self.counter.truncate(item.text, allowed)
        included_tokens = self.counter.count(selected_text)
        clean_metadata["included_passage_ids"] = selected_ids
        return replace(
            item,
            text=selected_text,
            content_mode="selected_passages",
            original_tokens=original_tokens,
            included_tokens=included_tokens,
            is_truncated=True,
            omission_reason="context_budget",
            metadata=clean_metadata,
        )

    def _select_passages(
        self, item: Evidence, passages: Any, allowed: int
    ) -> tuple[str, tuple[str, ...]]:
        if not isinstance(passages, list):
            return "", ()
        matched = set(item.matched_passage_ids)
        normalized = [passage for passage in passages if isinstance(passage, dict)]
        normalized.sort(
            key=lambda passage: (
                str(passage.get("passage_id")) not in matched,
                str(passage.get("segment_type")) != "decision",
                int(passage.get("sequence_index") or 0),
            )
        )
        chunks: list[str] = []
        ids: list[str] = []
        for passage in normalized:
            passage_id = str(passage.get("passage_id") or "")
            segment_type = str(passage.get("segment_type") or "passage")
            text = str(passage.get("text") or "").strip()
            if not passage_id or not text:
                continue
            block = f"[{segment_type} | {passage_id}]\n{text}"
            candidate = "\n\n".join([*chunks, block])
            if self.counter.count(candidate) > allowed:
                continue
            chunks.append(block)
            ids.append(passage_id)
        return "\n\n".join(chunks), tuple(ids)
