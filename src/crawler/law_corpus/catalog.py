from __future__ import annotations

from pathlib import Path
import tomllib

from crawler.law_corpus.models import AcquisitionSource


def load_sources(path: str | Path) -> list[AcquisitionSource]:
    catalog_path = Path(path)
    payload = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    raw_sources = payload.get("sources", [])
    sources: list[AcquisitionSource] = []
    for item in raw_sources:
        sources.append(
            AcquisitionSource(
                doc_id=item["doc_id"],
                title=item["title"],
                jurisdiction=item["jurisdiction"],
                law_family=item["law_family"],
                source_type=item["source_type"],
                version_date=item.get("version_date", ""),
                effective_date=item.get("effective_date", ""),
                language=item["language"],
                url=item["url"],
                preferred_format=item["preferred_format"],
                download_mode=item["download_mode"],
                target_path=item["target_path"],
                manual_instructions=item.get("manual_instructions", ""),
                source_set=item.get("source_set", "seed"),
                translation_status=item.get("translation_status", "official_original"),
            )
        )
    return sources
