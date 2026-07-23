from __future__ import annotations

from crawler.law_corpus.parsers.base import LegalParser
from crawler.law_corpus.parsers.china import ChinaLegalParser
from crawler.law_corpus.parsers.commonwealth import CommonwealthLegalParser
from crawler.law_corpus.parsers.eu import EuLegalParser
from crawler.law_corpus.parsers.us import UsLegalParser


PARSERS: list[LegalParser] = [
    EuLegalParser(),
    ChinaLegalParser(),
    UsLegalParser(),
    CommonwealthLegalParser(),
]


def get_parser(law_family: str) -> LegalParser:
    for parser in PARSERS:
        if law_family in parser.source_families:
            return parser
    raise ValueError(f"No legal parser registered for law_family={law_family}")
