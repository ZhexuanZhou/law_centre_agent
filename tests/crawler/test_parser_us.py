from crawler.law_corpus.models import SourceDocument
from crawler.law_corpus.parsers.us import UsLegalParser


def test_us_parser_extracts_cfr_sections():
    doc = SourceDocument(
        doc_id="us_hipaa_45_cfr_part_164",
        jurisdiction="US",
        law_family="us_cfr",
        source_type="primary_law",
        title="45 CFR Part 164",
        version_date="2026-06-04",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "PART 164-SECURITY AND PRIVACY\n"
            "§ 164.502 Uses and disclosures of protected health information.\n"
            "(a) Standard. A covered entity may not use or disclose protected health information.\n"
            "§ 164.506 Uses and disclosures to carry out treatment, payment, or health care operations.\n"
            "(a) Standard. A covered entity may use or disclose protected health information.\n"
        ),
    )

    units = UsLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["§ 164.502", "§ 164.506"]
    assert units[0].canonical_citation == "45 CFR § 164.502"
    assert "may not use or disclose" in units[0].text


def test_us_parser_extracts_california_code_section():
    doc = SourceDocument(
        doc_id="us_ccpa_cpra_1798_100",
        jurisdiction="US-CA",
        law_family="us_state_privacy",
        source_type="primary_law",
        title="California Civil Code Section 1798.100",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Civil Code - CIV\n"
            "1798.100.\n"
            "General Duties of Businesses that Collect Personal Information\n"
            "(a) A business that controls the collection of a consumer's personal information shall inform consumers.\n"
        ),
    )

    units = UsLegalParser().parse(doc)

    assert len(units) == 1
    assert units[0].local_citation == "1798.100"
    assert units[0].canonical_citation == "California Civil Code § 1798.100"
    assert "General Duties" in units[0].text


def test_us_parser_extracts_multiple_california_code_sections():
    doc = SourceDocument(
        doc_id="us_ca_ccpa_cpra_civ_1798_100_199",
        jurisdiction="US-CA",
        law_family="us_state_privacy",
        source_type="primary_law",
        title="California Consumer Privacy Act and CPRA Amendments",
        version_date="current",
        effective_date=None,
        source_url="https://leginfo.legislature.ca.gov/",
        language="en",
        raw_text=(
            "1798.100.\n"
            "General Duties of Businesses that Collect Personal Information\n"
            "(a) A business shall inform consumers.\n"
            "1798.105.\n"
            "Consumers' Right to Delete Personal Information\n"
            "(a) A consumer shall have the right to request deletion.\n"
        ),
    )

    units = UsLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["1798.100", "1798.105"]
    assert "request deletion" in units[1].text


def test_us_parser_ignores_split_california_section_cross_references():
    doc = SourceDocument(
        doc_id="us_ca_ccpa_cpra_civ_1798_100_199",
        jurisdiction="US-CA",
        law_family="us_state_privacy",
        source_type="primary_law",
        title="California Consumer Privacy Act and CPRA Amendments",
        version_date="current",
        effective_date=None,
        source_url="https://leginfo.legislature.ca.gov/",
        language="en",
        raw_text=(
            "1798.160.\n"
            "Consumer Privacy Fund\n"
            "(a) A special fund is created.\n"
            "1798.199.55.\n"
            "Administrative enforcement\n"
            "(a) The agency may bring an administrative action pursuant to Section\n"
            "1798.160.\n"
            "(b) If two or more persons are responsible, they are jointly liable.\n"
            "1798.199.57.\n"
            "(a) A section can start directly with a subdivision marker.\n"
            "1798.199.60.\n"
            "Agency decision review\n"
            "Whenever the agency rejects the decision, it shall state the reasons.\n"
        ),
    )

    units = UsLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "1798.160",
        "1798.199.55",
        "1798.199.57",
        "1798.199.60",
    ]
    assert "jointly liable" in units[1].text
    assert "subdivision marker" in units[2].text
