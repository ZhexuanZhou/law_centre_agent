from __future__ import annotations

import math
import re
from typing import Sequence

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?；;])(?:\s+|(?=[\u3400-\u9fff]))")


class TokenCounter:
    """Deterministically approximate Qwen token usage without a network dependency."""

    def __init__(self, *, safety_factor: float = 1.2) -> None:
        if safety_factor < 1.0:
            raise ValueError("safety_factor must be at least 1.0")
        self.safety_factor = safety_factor

    def count(self, text: str) -> int:
        return math.ceil(_base_units(text) * self.safety_factor)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 1:
            return ""
        if self.count(text) <= max_tokens:
            return text
        marker = "\n[…]\n"
        raw_limit = max(1.0, max_tokens / self.safety_factor)
        available = max(raw_limit - _base_units(marker), 1.0)
        head_end = _end_for_units(text, 0, available * 2 / 3)
        tail_start = _start_for_units(text, len(text), available / 3)
        return f"{text[:head_end]}{marker}{text[tail_start:]}"

    def split_structured(
        self,
        text: str,
        *,
        target_tokens: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        if not text.strip():
            return []
        if not 0 <= overlap_tokens < target_tokens <= max_tokens:
            raise ValueError("chunk token limits must satisfy 0 <= overlap < target <= max")
        if self.count(text) <= max_tokens:
            return [text.strip()]

        paragraphs = [item.strip() for item in _PARAGRAPH_SPLIT_RE.split(text) if item.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
        atoms: list[str] = []
        for paragraph in paragraphs:
            if self.count(paragraph) <= max_tokens:
                atoms.append(paragraph)
            else:
                atoms.extend(
                    self._token_windows(
                        paragraph,
                        target_tokens=target_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )
        return self._pack_atoms(atoms, target_tokens=target_tokens, max_tokens=max_tokens)

    def _token_windows(self, text: str, *, target_tokens: int, overlap_tokens: int) -> list[str]:
        raw_target = max(1.0, target_tokens / self.safety_factor)
        raw_overlap = min(max(0.0, overlap_tokens / self.safety_factor), raw_target - 0.25)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = _end_for_units(text, start, raw_target)
            if end <= start:
                end = start + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            next_start = _start_for_units(text, end, raw_overlap)
            start = max(next_start, start + 1)
        return [chunk for chunk in chunks if chunk]

    def _pack_atoms(
        self, atoms: Sequence[str], *, target_tokens: int, max_tokens: int
    ) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        for atom in atoms:
            candidate = "\n\n".join([*current, atom])
            if current and self.count(candidate) > target_tokens:
                chunks.append("\n\n".join(current))
                current = [atom]
            else:
                current.append(atom)
            if self.count("\n\n".join(current)) >= max_tokens:
                chunks.append("\n\n".join(current))
                current = []
        if current:
            chunks.append("\n\n".join(current))
        return chunks


def _base_units(text: str) -> float:
    return sum(_character_units(character) for character in text)


def _character_units(character: str) -> float:
    codepoint = ord(character)
    if character.isspace():
        return 0.1
    if character.isascii():
        return 0.25 if character.isalnum() or character == "_" else 0.5
    if 0x3400 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
        return 1.0
    return 1.0


def _end_for_units(text: str, start: int, budget: float) -> int:
    used = 0.0
    end = start
    while end < len(text):
        cost = _character_units(text[end])
        if end > start and used + cost > budget:
            break
        used += cost
        end += 1
    return end


def _start_for_units(text: str, end: int, budget: float) -> int:
    used = 0.0
    start = end
    while start > 0:
        cost = _character_units(text[start - 1])
        if start < end and used + cost > budget:
            break
        used += cost
        start -= 1
    return start
