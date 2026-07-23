from crawler.law_corpus.models import SourceDocument
from crawler.law_corpus.parsers.eu import EuLegalParser


def test_eu_parser_extracts_articles():
    doc = SourceDocument(
        doc_id="eu_gdpr_2016_679",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Article 5\n"
            "Principles relating to processing of personal data\n"
            "1. Personal data shall be processed lawfully.\n"
            "(a) lawfulness, fairness and transparency;\n"
            "Article 6\n"
            "Lawfulness of processing\n"
            "1. Processing shall be lawful only if...\n"
        ),
    )

    units = EuLegalParser().parse(doc)

    article_units = [unit for unit in units if unit.unit_type == "article"]
    assert [unit.local_citation for unit in article_units] == ["Article 5", "Article 6"]
    assert article_units[0].canonical_citation == "GDPR Article 5"
    assert "Personal data shall be processed lawfully" in article_units[0].text
    assert article_units[0].span_ids[0].startswith("eu_gdpr_2016_679:span:")


def test_eu_parser_extracts_recitals_from_preamble():
    doc = SourceDocument(
        doc_id="eu_gdpr_2016_679",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Whereas:\n"
            "(1)\n"
            "The protection of natural persons is a fundamental right.\n"
            "(2)\n"
            "The principles should respect fundamental rights.\n"
            "HAVE ADOPTED THIS REGULATION:\n"
            "CHAPTER I\n"
            "Article 1\n"
            "Subject-matter and objectives\n"
            "1. This Regulation lays down rules.\n"
        ),
    )

    units = EuLegalParser().parse(doc)

    recital_units = [unit for unit in units if unit.unit_type == "recital"]
    assert [unit.local_citation for unit in recital_units] == ["Recital 1", "Recital 2"]
    assert recital_units[0].canonical_citation == "GDPR Recital 1"
    assert "fundamental right" in recital_units[0].text


def test_eu_parser_extracts_article_paragraphs_and_points():
    doc = SourceDocument(
        doc_id="eu_gdpr_2016_679",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Article 9\n"
            "Processing of special categories of personal data\n"
            "1.\n"
            "Processing of personal data revealing racial origin shall be prohibited.\n"
            "2.\n"
            "Paragraph 1 shall not apply if one of the following applies:\n"
            "(a)\n"
            "the data subject has given explicit consent;\n"
            "(b)\n"
            "processing is necessary for employment law obligations.\n"
            "Article 10\n"
            "Processing of criminal conviction data\n"
            "Processing shall be carried out only under official authority.\n"
        ),
    )

    units = EuLegalParser().parse(doc)
    by_citation = {unit.local_citation: unit for unit in units}

    assert by_citation["Article 9"].unit_type == "article"
    assert by_citation["Article 9(1)"].unit_type == "paragraph"
    assert by_citation["Article 9(2)"].parent_id == by_citation["Article 9"].unit_id
    assert by_citation["Article 9(2)(a)"].unit_type == "point"
    assert by_citation["Article 9(2)(a)"].parent_id == by_citation["Article 9(2)"].unit_id
    assert "explicit consent" in by_citation["Article 9(2)(a)"].text


def test_eu_parser_extracts_parenthesized_definition_items():
    doc = SourceDocument(
        doc_id="eu_gdpr_2016_679",
        jurisdiction="EU",
        law_family="eu_gdpr",
        source_type="primary_law",
        title="GDPR",
        version_date="2016-04-27",
        effective_date="2018-05-25",
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Article 4\n"
            "Definitions\n"
            "For the purposes of this Regulation:\n"
            "(1)\n"
            "'personal data' means any information relating to an identified person;\n"
            "(7)\n"
            "'controller' means the natural or legal person which determines purposes;\n"
            "(11)\n"
            "'consent' means any freely given indication of the data subject's wishes;\n"
            "(23)\n"
            "'cross-border processing' means either:\n"
            "(a)\n"
            "processing in more than one Member State; or\n"
            "(b)\n"
            "processing which substantially affects data subjects in more than one Member State;\n"
            "(24)\n"
            "'relevant and reasoned objection' means an objection to a draft decision.\n"
            "Article 5\n"
            "Principles relating to processing of personal data\n"
            "1. Personal data shall be processed lawfully.\n"
        ),
    )

    units = EuLegalParser().parse(doc)
    by_citation = {unit.local_citation: unit for unit in units}

    assert by_citation["Article 4(1)"].unit_type == "paragraph"
    assert by_citation["Article 4(7)"].parent_id == by_citation["Article 4"].unit_id
    assert by_citation["Article 4(11)"].canonical_citation == "GDPR Article 4(11)"
    assert by_citation["Article 4(24)"].text.startswith("(24)")
    assert by_citation["Article 4(23)(a)"].unit_type == "point"
    assert by_citation["Article 4(23)(b)"].parent_id == by_citation["Article 4(23)"].unit_id
    assert by_citation["Article 5(1)"].unit_type == "paragraph"


def test_eu_parser_keeps_nested_roman_points_under_parent_letter_point():
    doc = SourceDocument(
        doc_id="eu_ai_act_test",
        jurisdiction="EU",
        law_family="eu_ai_act",
        source_type="primary_law",
        title="EU AI Act",
        version_date="2024-06-13",
        effective_date="2024-08-01",
        source_url="https://example.test/ai-act",
        language="en",
        raw_text=(
            "HAVE ADOPTED THIS REGULATION:\n"
            "Article 5\n"
            "Prohibited AI practices\n"
            "1. The following AI practices shall be prohibited:\n"
            "(c)\n"
            "social scoring leading to either or both of the following:\n"
            "(i)\n"
            "unrelated detrimental treatment;\n"
            "(ii)\n"
            "disproportionate detrimental treatment;\n"
            "(d)\n"
            "criminal risk assessment based solely on profiling;\n"
            "(h)\n"
            "real-time remote biometric identification unless necessary for one of the following objectives:\n"
            "(i)\n"
            "targeted search for victims;\n"
            "(ii)\n"
            "prevention of imminent threats;\n"
            "(iii)\n"
            "identification of a serious-crime suspect.\n"
            "2. Paragraph 2 text.\n"
            "(a)\n"
            "first element for paragraph 2.\n"
            "Article 6\n"
            "Classification rules\n"
            "1. High-risk systems are classified by this Article.\n"
        ),
    )

    units = EuLegalParser().parse(doc)
    by_citation = {unit.local_citation: unit for unit in units}

    assert len({unit.unit_id for unit in units}) == len(units)
    assert by_citation["Article 5(1)(c)"].parent_id == by_citation["Article 5(1)"].unit_id
    assert by_citation["Article 5(1)(c)(i)"].parent_id == by_citation["Article 5(1)(c)"].unit_id
    assert "unrelated detrimental treatment" in by_citation["Article 5(1)(c)(i)"].text
    assert by_citation["Article 5(1)(h)(i)"].parent_id == by_citation["Article 5(1)(h)"].unit_id
    assert by_citation["Article 5(1)(h)(iii)"].canonical_citation == (
        "EU AI Act Article 5(1)(h)(iii)"
    )
    assert "Article 5(1)(i)" not in by_citation
    assert by_citation["Article 5(2)(a)"].parent_id == by_citation["Article 5(2)"].unit_id


def test_eu_parser_does_not_treat_line_start_article_references_as_headings():
    doc = SourceDocument(
        doc_id="eu_ai_act_test",
        jurisdiction="EU",
        law_family="eu_ai_act",
        source_type="primary_law",
        title="EU AI Act",
        version_date="2024-06-13",
        effective_date="2024-08-01",
        source_url="https://example.test/ai-act",
        language="en",
        raw_text=(
            "HAVE ADOPTED THIS REGULATION:\n"
            "Article 6\n"
            "Classification rules\n"
            "1. The high-risk classification rule applies.\n"
            "Article 7\n"
            "Amendments\n"
            "1. The Commission may adopt delegated acts.\n"
            "Article 6(1) and the corresponding obligations shall apply later.\n"
            "Article 18 of Regulation (EU) 2019/1020 shall apply mutatis mutandis.\n"
            "Article 8\n"
            "Compliance\n"
            "1. Providers shall comply with requirements.\n"
        ),
    )

    units = EuLegalParser().parse(doc)
    articles = [unit for unit in units if unit.unit_type == "article"]

    assert [unit.local_citation for unit in articles] == [
        "Article 6",
        "Article 7",
        "Article 8",
    ]
    article_7 = next(unit for unit in articles if unit.local_citation == "Article 7")
    assert "Article 6(1) and the corresponding obligations" in article_7.text
    assert "Article 18 of Regulation" in article_7.text


def test_eu_parser_excludes_signature_and_footnotes_from_final_article():
    doc = SourceDocument(
        doc_id="eu_ai_act_test",
        jurisdiction="EU",
        law_family="eu_ai_act",
        source_type="primary_law",
        title="EU AI Act",
        version_date="2024-06-13",
        effective_date="2024-08-01",
        source_url="https://example.test/ai-act",
        language="en",
        raw_text=(
            "HAVE ADOPTED THIS REGULATION:\n"
            "Article 113\n"
            "Entry into force and application\n"
            "This Regulation shall enter into force on the twentieth day.\n"
            "This Regulation shall be binding in its entirety.\n"
            "Done at Brussels, 13 June 2024.\n"
            "For the European Parliament\n"
            "The President\n"
            "R. METSOLA\n"
            "(\n"
            "1\n"
            ")\n"
            "OJ C 517, 22.12.2021, p. 56.\n"
        ),
    )

    units = EuLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 113"]
    assert "Done at Brussels" not in units[0].text
    assert "OJ C 517" not in units[0].text


def test_eu_parser_excludes_annex_correlation_table_article_references():
    doc = SourceDocument(
        doc_id="eu_nis2_directive_2022_2555",
        jurisdiction="EU",
        law_family="eu_nis2",
        source_type="primary_law",
        title="Directive (EU) 2022/2555 NIS2 Directive",
        version_date="2022-12-14",
        effective_date="2023-01-16",
        source_url="https://eur-lex.europa.eu/eli/dir/2022/2555/oj",
        language="en",
        raw_text=(
            "HAVE ADOPTED THIS DIRECTIVE:\n"
            "Article 4\n"
            "Sector-specific Union legal acts\n"
            "1. This Directive shall not apply where sector-specific acts impose equivalent requirements.\n"
            "Article 5\n"
            "Minimum harmonisation\n"
            "This Directive shall not preclude Member States from adopting stronger rules.\n"
            "ANNEX III\n"
            "CORRELATION TABLE\n"
            "Article 4\n"
            "Article 2\n"
            "Article 5\n"
            "Article 4\n"
        ),
    )

    units = EuLegalParser().parse(doc)
    article_units = [unit for unit in units if unit.unit_type == "article"]

    assert [unit.local_citation for unit in article_units] == ["Article 4", "Article 5"]
    assert "Sector-specific Union legal acts" in article_units[0].text
    assert "CORRELATION TABLE" not in article_units[1].text


def test_eu_parser_uses_specific_labels_for_data_laws():
    doc = SourceDocument(
        doc_id="eu_data_act_2023_2854",
        jurisdiction="EU",
        law_family="eu_data_act",
        source_type="primary_law",
        title="Regulation (EU) 2023/2854 Data Act",
        version_date="2023-12-13",
        effective_date="2024-01-11",
        source_url="https://eur-lex.europa.eu/eli/reg/2023/2854/oj",
        language="en",
        raw_text=(
            "Whereas:\n"
            "(1)\n"
            "Data is a core component of the digital economy.\n"
            "HAVE ADOPTED THIS REGULATION:\n"
            "Article 1\n"
            "Subject matter and scope\n"
            "1. This Regulation lays down harmonised rules.\n"
        ),
    )

    units = EuLegalParser().parse(doc)

    assert units[0].canonical_citation == "EU Data Act Recital 1"
    assert units[1].canonical_citation == "EU Data Act Article 1"
