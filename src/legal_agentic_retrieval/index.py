from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata

import numpy as np

from legal_agentic_retrieval.models import Evidence, ExactCitation, RetrievalFilters
from legal_agentic_retrieval.providers import Embedder
from legal_agentic_retrieval.tokenization import TokenCounter


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _VectorDocument:
    evidence_id: str
    source_type: str
    record_id: str
    text: str
    jurisdiction: str | None
    country: str | None
    doc_id: str | None
    decided_date: str | None


@dataclass(frozen=True)
class _Passage:
    passage_id: str
    parent_evidence_id: str
    source_type: str
    segment_type: str
    sequence_index: int
    text: str
    token_count: int
    embedding_text: str


class CorpusIndexBuilder:
    """Project the structured corpus into exact records and searchable embeddings."""

    def __init__(
        self,
        corpus_dir: str | Path,
        embedder: Embedder,
        *,
        token_counter: TokenCounter | None = None,
        passage_threshold_tokens: int = 1_600,
        passage_target_tokens: int = 800,
        passage_max_tokens: int = 1_000,
        passage_overlap_tokens: int = 100,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.embedder = embedder
        self.token_counter = token_counter or TokenCounter()
        self.passage_threshold_tokens = passage_threshold_tokens
        self.passage_target_tokens = passage_target_tokens
        self.passage_max_tokens = passage_max_tokens
        self.passage_overlap_tokens = passage_overlap_tokens

    def build(self, output_path: str | Path) -> dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        document_vector_output = document_vector_path_for(output)
        passage_vector_output = passage_vector_path_for(output)
        token = f"{os.getpid()}"
        temporary_db = output.with_name(f".{output.name}.{token}.tmp")
        temporary_document_vectors = output.with_name(f".{document_vector_output.name}.{token}.tmp")
        temporary_passage_vectors = output.with_name(f".{passage_vector_output.name}.{token}.tmp")
        temporary_paths = (
            temporary_db,
            temporary_document_vectors,
            temporary_passage_vectors,
        )
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        try:
            stats, vector_documents, passages = self._build_metadata(temporary_db)
            self._build_vectors(temporary_document_vectors, vector_documents)
            self._build_vectors(
                temporary_passage_vectors,
                [
                    _VectorDocument(
                        evidence_id=item.passage_id,
                        source_type=item.source_type,
                        record_id=item.passage_id,
                        text=item.embedding_text,
                        jurisdiction=None,
                        country=None,
                        doc_id=None,
                        decided_date=None,
                    )
                    for item in passages
                ],
            )
            os.replace(temporary_document_vectors, document_vector_output)
            os.replace(temporary_passage_vectors, passage_vector_output)
            os.replace(temporary_db, output)
        finally:
            for path in temporary_paths:
                if path.exists():
                    path.unlink()
        return {
            **stats,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "index": str(output),
            "document_vector_file": str(document_vector_output),
            "passage_vector_file": str(passage_vector_output),
        }

    def _build_metadata(
        self, path: Path
    ) -> tuple[dict[str, int], list[_VectorDocument], list[_Passage]]:
        laws = list(_read_jsonl(self.corpus_dir / "laws.jsonl"))
        units = [
            item
            for item in _read_jsonl(self.corpus_dir / "legal_units.jsonl")
            if item.get("is_current") is True
        ]
        parent_ids = {str(item["parent_id"]) for item in units if item.get("parent_id")}
        cases = [
            item
            for item in _read_jsonl(self.corpus_dir / "gdprhub_cases.jsonl")
            if str(item.get("facts_text") or "").strip()
            or str(item.get("decision_text") or "").strip()
        ]
        vector_documents: list[_VectorDocument] = []
        passages: list[_Passage] = []
        with sqlite3.connect(path) as connection:
            _create_schema(connection)
            for law in laws:
                connection.execute(
                    "INSERT INTO laws VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        law["doc_id"],
                        law["title"],
                        law.get("jurisdiction"),
                        law.get("language"),
                        law.get("law_family"),
                        law.get("source_url"),
                        law.get("version_date"),
                        law.get("effective_date"),
                    ),
                )
            for unit in units:
                is_leaf = str(unit["unit_id"]) not in parent_ids
                connection.execute(
                    "INSERT INTO law_units VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        unit["unit_id"],
                        unit["source_doc_id"],
                        unit.get("parent_id"),
                        unit["unit_type"],
                        unit["canonical_citation"],
                        unit["local_citation"],
                        _match_key(str(unit["local_citation"])),
                        unit["text"],
                        unit.get("jurisdiction"),
                        unit.get("law_name"),
                        unit.get("effective_from"),
                        unit.get("effective_to"),
                    ),
                )
                if is_leaf:
                    law = next(item for item in laws if item["doc_id"] == unit["source_doc_id"])
                    evidence_id = f"law_unit:{unit['unit_id']}"
                    vector_documents.append(
                        _VectorDocument(
                            evidence_id=evidence_id,
                            source_type="law_unit",
                            record_id=str(unit["unit_id"]),
                            text=_law_embedding_text(law, unit),
                            jurisdiction=_optional_text(unit.get("jurisdiction")),
                            country=None,
                            doc_id=str(unit["source_doc_id"]),
                            decided_date=None,
                        )
                    )
                    passages.extend(
                        self._make_passages(
                            parent_evidence_id=evidence_id,
                            source_type="law_unit",
                            header="\n".join(
                                str(value)
                                for value in (
                                    law.get("title"),
                                    unit.get("canonical_citation"),
                                    unit.get("jurisdiction"),
                                )
                                if value
                            ),
                            segments=[
                                (str(unit.get("unit_type") or "legal_unit"), str(unit["text"]))
                            ],
                        )
                    )
            case_ids: set[str] = set()
            for case in cases:
                case_id = str(case["case_id"])
                case_ids.add(case_id)
                industries = [
                    str(item.get("industry"))
                    for item in case.get("industry") or []
                    if isinstance(item, Mapping) and item.get("industry")
                ]
                connection.execute(
                    "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        case_id,
                        case["title"],
                        case.get("authority"),
                        case.get("jurisdiction"),
                        case.get("country"),
                        case.get("decided_date"),
                        case.get("case_number"),
                        case.get("ecli"),
                        case.get("company_or_parties"),
                        case.get("facts_text"),
                        case.get("decision_text"),
                        _json_text(case.get("outcome")),
                        case.get("source_url"),
                        case.get("original_source_url"),
                        json.dumps(industries, ensure_ascii=False),
                        json.dumps(case.get("categories") or [], ensure_ascii=False),
                    ),
                )
                vector_documents.append(
                    _VectorDocument(
                        evidence_id=f"case:{case_id}",
                        source_type="case",
                        record_id=case_id,
                        text=_case_embedding_text(case, industries),
                        jurisdiction=_optional_text(case.get("jurisdiction")),
                        country=_optional_text(case.get("country")),
                        doc_id=None,
                        decided_date=_optional_text(case.get("decided_date")),
                    )
                )
                passages.extend(
                    self._make_passages(
                        parent_evidence_id=f"case:{case_id}",
                        source_type="case",
                        header="\n".join(
                            str(value)
                            for value in (
                                case.get("title"),
                                case.get("authority"),
                                case.get("country"),
                            )
                            if value
                        ),
                        segments=[
                            ("facts", str(case.get("facts_text") or "")),
                            ("decision", str(case.get("decision_text") or "")),
                        ],
                    )
                )
            unit_ids = {str(item["unit_id"]) for item in units}
            case_law_relations = 0
            for relation in _read_jsonl(self.corpus_dir / "case_law_relations.jsonl"):
                unit_id = relation.get("target_unit_id")
                case_id = str(relation["case_id"])
                if (
                    relation.get("resolution_status") != "resolved"
                    or case_id not in case_ids
                    or not unit_id
                    or str(unit_id) not in unit_ids
                ):
                    continue
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO case_law_relations VALUES (?, ?, ?)",
                    (case_id, unit_id, relation.get("citation")),
                )
                case_law_relations += max(cursor.rowcount, 0)
            cross_law_relations = 0
            for relation in _read_jsonl(self.corpus_dir / "law_relations.jsonl"):
                source_id = str(relation.get("source_unit_id") or "")
                target_id = str(relation.get("target_unit_id") or "")
                if (
                    relation.get("relation_scope") != "cross_law"
                    or source_id not in unit_ids
                    or target_id not in unit_ids
                ):
                    continue
                evidence = relation.get("evidence") or []
                excerpt = evidence[0].get("text_excerpt") if evidence else None
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO cross_law_relations VALUES (?, ?, ?)",
                    (source_id, target_id, excerpt),
                )
                cross_law_relations += max(cursor.rowcount, 0)
            for position, document in enumerate(vector_documents):
                connection.execute(
                    "INSERT INTO vector_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        position,
                        document.evidence_id,
                        document.source_type,
                        document.record_id,
                        document.jurisdiction,
                        document.country,
                        document.doc_id,
                        document.decided_date,
                        _sha256(document.text),
                    ),
                )
            for position, passage in enumerate(passages):
                connection.execute(
                    "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        passage.passage_id,
                        passage.parent_evidence_id,
                        passage.source_type,
                        passage.segment_type,
                        passage.sequence_index,
                        passage.text,
                        passage.token_count,
                    ),
                )
                connection.execute(
                    "INSERT INTO passage_vectors VALUES (?, ?, ?, ?)",
                    (
                        position,
                        passage.passage_id,
                        passage.parent_evidence_id,
                        passage.source_type,
                    ),
                )
            metadata = {
                "schema_version": str(SCHEMA_VERSION),
                "embedding_model": self.embedder.model_name,
                "embedding_dimension": str(self.embedder.dimension),
                "document_vector_count": str(len(vector_documents)),
                "passage_vector_count": str(len(passages)),
                "passage_threshold_tokens": str(self.passage_threshold_tokens),
            }
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        return (
            {
                "laws": len(laws),
                "law_units": len(units),
                "embedded_leaf_law_units": sum(
                    item.source_type == "law_unit" for item in vector_documents
                ),
                "cases": len(cases),
                "case_law_relations": case_law_relations,
                "cross_law_relations": cross_law_relations,
                "document_vector_count": len(vector_documents),
                "passage_vector_count": len(passages),
            },
            vector_documents,
            passages,
        )

    def _make_passages(
        self,
        *,
        parent_evidence_id: str,
        source_type: str,
        header: str,
        segments: Sequence[tuple[str, str]],
    ) -> list[_Passage]:
        combined = "\n\n".join(text for _, text in segments if text.strip())
        if self.token_counter.count(combined) <= self.passage_threshold_tokens:
            return []
        output: list[_Passage] = []
        sequence_index = 0
        for segment_type, text in segments:
            for chunk in self.token_counter.split_structured(
                text,
                target_tokens=self.passage_target_tokens,
                max_tokens=self.passage_max_tokens,
                overlap_tokens=self.passage_overlap_tokens,
            ):
                passage_id = f"{parent_evidence_id}:passage:{segment_type}:{sequence_index}"
                output.append(
                    _Passage(
                        passage_id=passage_id,
                        parent_evidence_id=parent_evidence_id,
                        source_type=source_type,
                        segment_type=segment_type,
                        sequence_index=sequence_index,
                        text=chunk,
                        token_count=self.token_counter.count(chunk),
                        embedding_text=f"{header}\n{segment_type}\n{chunk}",
                    )
                )
                sequence_index += 1
        return output

    def _build_vectors(self, path: Path, documents: Sequence[_VectorDocument]) -> None:
        if not documents:
            with path.open("wb") as handle:
                np.save(
                    handle,
                    np.empty((0, self.embedder.dimension), dtype=np.float32),
                )
            return
        matrix = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(documents), self.embedder.dimension),
        )
        for start in range(0, len(documents), self.embedder.batch_size):
            batch = documents[start : start + self.embedder.batch_size]
            vectors = self.embedder.embed([item.text for item in batch])
            if vectors.shape != (len(batch), self.embedder.dimension):
                raise ValueError("embedder returned an unexpected matrix shape")
            matrix[start : start + len(batch)] = vectors
        matrix.flush()


class RetrievalIndex:
    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self.path = Path(path)
        self.document_vector_path = document_vector_path_for(self.path)
        self.passage_vector_path = passage_vector_path_for(self.path)
        self.embedder = embedder
        if not all(
            path.is_file()
            for path in (self.path, self.document_vector_path, self.passage_vector_path)
        ):
            raise FileNotFoundError(
                "SQLite, .document_vectors.npy, and .passage_vectors.npy files are required"
            )
        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        if int(metadata.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"index schema mismatch: expected {SCHEMA_VERSION}, "
                f"received {metadata.get('schema_version')}"
            )
        expected = int(metadata["embedding_dimension"])
        if expected != embedder.dimension:
            raise ValueError(
                f"embedding dimension mismatch: index={expected}, provider={embedder.dimension}"
            )
        if metadata["embedding_model"] != embedder.model_name:
            raise ValueError(
                "embedding model mismatch: "
                f"index={metadata['embedding_model']}, provider={embedder.model_name}"
            )
        self.document_vectors = np.load(self.document_vector_path, mmap_mode="r")
        passage_count = int(metadata["passage_vector_count"])
        self.passage_vectors = np.load(
            self.passage_vector_path,
            mmap_mode="r" if passage_count else None,
        )

    def catalog(self) -> dict[str, Any]:
        with self._connect() as connection:
            law_rows = connection.execute(
                "SELECT doc_id, title, jurisdiction, language, law_family FROM laws "
                "ORDER BY jurisdiction, title"
            ).fetchall()
            laws: list[dict[str, Any]] = []
            for law_row in law_rows:
                law = dict(law_row)
                examples = connection.execute(
                    "SELECT unit_type, min(local_citation) FROM law_units "
                    "WHERE doc_id = ? GROUP BY unit_type ORDER BY unit_type",
                    (law_row["doc_id"],),
                ).fetchall()
                law["citation_examples"] = {row[0]: row[1] for row in examples}
                laws.append(law)
            countries = connection.execute(
                "SELECT DISTINCT country FROM cases WHERE country IS NOT NULL ORDER BY country"
            ).fetchall()
            case_jurisdictions = connection.execute(
                "SELECT DISTINCT jurisdiction FROM cases WHERE jurisdiction IS NOT NULL "
                "ORDER BY jurisdiction"
            ).fetchall()
            dates = connection.execute(
                "SELECT min(decided_date), max(decided_date) FROM cases "
                "WHERE decided_date IS NOT NULL"
            ).fetchone()
        return {
            "laws": laws,
            "available_case_countries": [row[0] for row in countries],
            "available_case_jurisdictions": [row[0] for row in case_jurisdictions],
            "case_date_range": [dates[0], dates[1]],
        }

    def exact(self, citations: Sequence[ExactCitation]) -> list[Evidence]:
        evidence: list[Evidence] = []
        with self._connect() as connection:
            for citation in citations:
                rows = connection.execute(
                    "SELECT u.*, l.title, l.source_url FROM law_units u "
                    "JOIN laws l ON l.doc_id = u.doc_id "
                    "WHERE u.doc_id = ? AND u.local_citation_key = ?",
                    (citation.doc_id, _match_key(citation.local_citation)),
                ).fetchall()
                evidence.extend(_law_row(row, score=1.0) for row in rows)
        return _dedupe(evidence)

    def vector_search(
        self,
        queries: Sequence[str],
        filters: RetrievalFilters,
        *,
        limit: int,
    ) -> list[Evidence]:
        usable_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not usable_queries:
            return []
        query_vectors = self.embedder.embed(usable_queries)
        with self._connect() as connection:
            rows = self._candidate_rows(connection, filters)
            if not rows:
                return []
            positions = np.asarray([row["vector_position"] for row in rows], dtype=np.int64)
            candidate_vectors = np.asarray(self.document_vectors[positions], dtype=np.float32)
            scores = np.max(candidate_vectors @ query_vectors.T, axis=1)
            order = _diverse_order(rows, scores, limit)
            evidence = [
                self._evidence_from_vector_row(connection, rows[int(index)], float(scores[index]))
                for index in order
            ]
            passage_matches = self._passage_matches(
                connection,
                [item.evidence_id for item in evidence],
                query_vectors,
                limit=max(limit * 3, 30),
            )
            return [
                replace(
                    item,
                    score=max(
                        item.score,
                        max(
                            (score for _, score in passage_matches.get(item.evidence_id, [])),
                            default=item.score,
                        ),
                    ),
                    matched_passage_ids=tuple(
                        passage_id for passage_id, _ in passage_matches.get(item.evidence_id, [])
                    ),
                )
                for item in evidence
            ]

    def _passage_matches(
        self,
        connection: sqlite3.Connection,
        parent_ids: Sequence[str],
        query_vectors: np.ndarray,
        *,
        limit: int,
    ) -> dict[str, list[tuple[str, float]]]:
        if not parent_ids or not len(self.passage_vectors):
            return {}
        rows = connection.execute(
            "SELECT * FROM passage_vectors "
            f"WHERE parent_evidence_id IN ({_placeholders(parent_ids)})",
            parent_ids,
        ).fetchall()
        if not rows:
            return {}
        positions = np.asarray([row["vector_position"] for row in rows], dtype=np.int64)
        vectors = np.asarray(self.passage_vectors[positions], dtype=np.float32)
        scores = np.max(vectors @ query_vectors.T, axis=1)
        order = np.argsort(-scores)[:limit]
        matches: dict[str, list[tuple[str, float]]] = {}
        for index in order:
            row = rows[int(index)]
            matches.setdefault(str(row["parent_evidence_id"]), []).append(
                (str(row["passage_id"]), float(scores[index]))
            )
        return matches

    def related_laws(self, case_ids: Sequence[str], *, limit: int) -> list[Evidence]:
        if not case_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT u.*, l.title, l.source_url FROM case_law_relations r "
                "JOIN law_units u ON u.unit_id = r.unit_id "
                "JOIN laws l ON l.doc_id = u.doc_id "
                f"WHERE r.case_id IN ({_placeholders(case_ids)}) LIMIT ?",
                (*case_ids, limit),
            ).fetchall()
        return [_law_row(row, score=1.0) for row in rows]

    def hydrate(self, evidence: Sequence[Evidence]) -> list[Evidence]:
        """Replace retrieval previews with complete records and their stored passages."""
        if not evidence:
            return []
        evidence_ids = [item.evidence_id for item in evidence]
        hydrated: list[Evidence] = []
        with self._connect() as connection:
            passage_rows = connection.execute(
                "SELECT * FROM passages "
                f"WHERE parent_evidence_id IN ({_placeholders(evidence_ids)}) "
                "ORDER BY parent_evidence_id, sequence_index",
                evidence_ids,
            ).fetchall()
            passages_by_parent: dict[str, list[dict[str, Any]]] = {}
            for row in passage_rows:
                passages_by_parent.setdefault(str(row["parent_evidence_id"]), []).append(
                    {
                        "passage_id": row["passage_id"],
                        "segment_type": row["segment_type"],
                        "sequence_index": row["sequence_index"],
                        "text": row["text"],
                        "token_count": row["token_count"],
                    }
                )
            for item in evidence:
                if item.source_type == "law_unit":
                    row = connection.execute(
                        "SELECT u.*, l.title, l.source_url FROM law_units u "
                        "JOIN laws l ON l.doc_id = u.doc_id WHERE u.unit_id = ?",
                        (item.metadata["unit_id"],),
                    ).fetchone()
                    full = _law_row(row, score=item.score, preview=False)
                else:
                    row = connection.execute(
                        "SELECT * FROM cases WHERE case_id = ?",
                        (item.metadata["case_id"],),
                    ).fetchone()
                    full = _case_row(row, score=item.score, preview=False)
                metadata = dict(full.metadata)
                metadata["passages"] = passages_by_parent.get(item.evidence_id, [])
                hydrated.append(
                    replace(
                        full,
                        metadata=metadata,
                        matched_passage_ids=item.matched_passage_ids,
                        content_mode="full",
                    )
                )
        return hydrated

    def _candidate_rows(
        self, connection: sqlite3.Connection, filters: RetrievalFilters
    ) -> list[sqlite3.Row]:
        clauses = [f"v.source_type IN ({_placeholders(filters.source_types)})"]
        parameters: list[Any] = list(filters.source_types)
        if filters.jurisdictions:
            clauses.append(
                "(v.source_type = 'case' OR "
                f"v.jurisdiction IN ({_placeholders(filters.jurisdictions)}))"
            )
            parameters.extend(filters.jurisdictions)
        if filters.countries:
            clauses.append(
                "(v.source_type = 'law_unit' OR "
                f"v.country IN ({_placeholders(filters.countries)}) OR "
                f"v.jurisdiction IN ({_placeholders(filters.countries)}))"
            )
            parameters.extend(filters.countries)
            parameters.extend(filters.countries)
        if filters.doc_ids:
            placeholders = _placeholders(filters.doc_ids)
            clauses.append(
                "((v.source_type = 'law_unit' AND v.doc_id IN ("
                f"{placeholders})) OR (v.source_type = 'case' AND EXISTS ("
                "SELECT 1 FROM case_law_relations r JOIN law_units u ON u.unit_id = r.unit_id "
                f"WHERE r.case_id = v.record_id AND u.doc_id IN ({placeholders}))))"
            )
            parameters.extend(filters.doc_ids)
            parameters.extend(filters.doc_ids)
        if filters.date_from:
            clauses.append("(v.decided_date IS NULL OR v.decided_date >= ?)")
            parameters.append(filters.date_from)
        if filters.date_to:
            clauses.append("(v.decided_date IS NULL OR v.decided_date <= ?)")
            parameters.append(filters.date_to)
        return connection.execute(
            f"SELECT v.* FROM vector_documents v WHERE {' AND '.join(clauses)}",
            parameters,
        ).fetchall()

    def _evidence_from_vector_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row, score: float
    ) -> Evidence:
        if row["source_type"] == "law_unit":
            law = connection.execute(
                "SELECT u.*, l.title, l.source_url FROM law_units u "
                "JOIN laws l ON l.doc_id = u.doc_id WHERE u.unit_id = ?",
                (row["record_id"],),
            ).fetchone()
            return _law_row(law, score=score)
        case = connection.execute(
            "SELECT * FROM cases WHERE case_id = ?", (row["record_id"],)
        ).fetchone()
        return _case_row(case, score=score)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE laws (
            doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, jurisdiction TEXT, language TEXT,
            law_family TEXT, source_url TEXT, version_date TEXT, effective_date TEXT
        );
        CREATE TABLE law_units (
            unit_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, parent_id TEXT, unit_type TEXT,
            canonical_citation TEXT, local_citation TEXT, local_citation_key TEXT, text TEXT,
            jurisdiction TEXT, law_name TEXT, effective_from TEXT, effective_to TEXT
        );
        CREATE INDEX law_units_exact_idx ON law_units(doc_id, local_citation_key);
        CREATE TABLE cases (
            case_id TEXT PRIMARY KEY, title TEXT, authority TEXT, jurisdiction TEXT, country TEXT,
            decided_date TEXT, case_number TEXT, ecli TEXT, company_or_parties TEXT,
            facts_text TEXT, decision_text TEXT, outcome TEXT, source_url TEXT,
            original_source_url TEXT, industries TEXT, categories TEXT
        );
        CREATE TABLE case_law_relations (
            case_id TEXT, unit_id TEXT, citation TEXT,
            PRIMARY KEY(case_id, unit_id, citation)
        );
        CREATE INDEX case_law_case_idx ON case_law_relations(case_id);
        CREATE TABLE cross_law_relations (
            source_unit_id TEXT, target_unit_id TEXT, evidence_excerpt TEXT,
            PRIMARY KEY(source_unit_id, target_unit_id)
        );
        CREATE TABLE vector_documents (
            vector_position INTEGER PRIMARY KEY, evidence_id TEXT UNIQUE, source_type TEXT,
            record_id TEXT, jurisdiction TEXT, country TEXT, doc_id TEXT, decided_date TEXT,
            text_sha256 TEXT
        );
        CREATE TABLE passages (
            passage_id TEXT PRIMARY KEY, parent_evidence_id TEXT NOT NULL,
            source_type TEXT NOT NULL, segment_type TEXT NOT NULL,
            sequence_index INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL
        );
        CREATE INDEX passages_parent_idx ON passages(parent_evidence_id, sequence_index);
        CREATE TABLE passage_vectors (
            vector_position INTEGER PRIMARY KEY, passage_id TEXT UNIQUE NOT NULL,
            parent_evidence_id TEXT NOT NULL, source_type TEXT NOT NULL
        );
        CREATE INDEX passage_vectors_parent_idx ON passage_vectors(parent_evidence_id);
        """
    )


def document_vector_path_for(index_path: str | Path) -> Path:
    path = Path(index_path)
    return path.with_suffix(".document_vectors.npy")


def passage_vector_path_for(index_path: str | Path) -> Path:
    path = Path(index_path)
    return path.with_suffix(".passage_vectors.npy")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _match_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _law_embedding_text(law: Mapping[str, Any], unit: Mapping[str, Any]) -> str:
    return "\n".join(
        str(value)
        for value in (
            law.get("title"),
            unit.get("jurisdiction"),
            unit.get("canonical_citation"),
            unit.get("text"),
        )
        if value
    )


def _case_embedding_text(case: Mapping[str, Any], industries: Sequence[str]) -> str:
    return "\n".join(
        str(value)
        for value in (
            case.get("title"),
            case.get("authority"),
            case.get("country"),
            " ".join(industries),
            " ".join(str(item) for item in case.get("categories") or []),
            case.get("facts_text"),
            case.get("decision_text"),
        )
        if value
    )


def _law_row(row: sqlite3.Row, *, score: float, preview: bool = True) -> Evidence:
    text = str(row["text"] or "")
    return Evidence(
        evidence_id=f"law_unit:{row['unit_id']}",
        source_type="law_unit",
        title=row["title"],
        text=_excerpt(text) if preview else text,
        score=score,
        source_url=row["source_url"],
        jurisdiction=row["jurisdiction"],
        citation=row["canonical_citation"],
        metadata={
            "unit_id": row["unit_id"],
            "doc_id": row["doc_id"],
            "unit_type": row["unit_type"],
            "effective_from": row["effective_from"],
            "effective_to": row["effective_to"],
        },
    )


def _case_row(row: sqlite3.Row, *, score: float, preview: bool = True) -> Evidence:
    facts_text = str(row["facts_text"] or "")
    decision_text = str(row["decision_text"] or "")
    facts = _excerpt(facts_text, 500) if preview else facts_text
    decision = _excerpt(decision_text, 500) if preview else decision_text
    return Evidence(
        evidence_id=f"case:{row['case_id']}",
        source_type="case",
        title=row["title"],
        text=f"Facts: {facts}\nDecision: {decision}",
        score=score,
        source_url=row["source_url"],
        jurisdiction=row["jurisdiction"],
        country=row["country"],
        citation=row["case_number"] or row["ecli"],
        metadata={
            "case_id": row["case_id"],
            "authority": row["authority"],
            "decided_date": row["decided_date"],
            "original_source_url": row["original_source_url"],
            "industries": json.loads(row["industries"] or "[]"),
            "outcome": row["outcome"],
        },
    )


def _excerpt(text: str, limit: int = 1000) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit]}…"


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _placeholders(values: Sequence[Any]) -> str:
    return ",".join("?" for _ in values)


def _dedupe(items: Iterable[Evidence]) -> list[Evidence]:
    return list({item.evidence_id: item for item in items}.values())


def _diverse_order(rows: Sequence[sqlite3.Row], scores: np.ndarray, limit: int) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["source_type"]), []).append(index)
    selected: list[int] = []
    per_group = max(1, limit // max(len(groups), 1))
    for indices in groups.values():
        ranked = sorted(indices, key=lambda index: float(scores[index]), reverse=True)
        selected.extend(ranked[:per_group])
    selected_set = set(selected)
    remaining = sorted(
        (index for index in range(len(rows)) if index not in selected_set),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    selected.extend(remaining[: max(limit - len(selected), 0)])
    return np.asarray(
        sorted(selected[:limit], key=lambda index: float(scores[index]), reverse=True),
        dtype=np.int64,
    )
