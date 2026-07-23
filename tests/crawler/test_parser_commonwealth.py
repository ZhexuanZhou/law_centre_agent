import pytest

from crawler.law_corpus.models import SourceDocument
from crawler.law_corpus.parsers.commonwealth import CommonwealthLegalParser


def test_commonwealth_parser_extracts_numbered_sections():
    doc = SourceDocument(
        doc_id="singapore_pdpa_2012",
        jurisdiction="SG",
        law_family="singapore_pdpa",
        source_type="primary_law",
        title="Personal Data Protection Act 2012",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "PART 3\n"
            "COLLECTION, USE AND DISCLOSURE OF PERSONAL DATA\n"
            "13 Consent required\n"
            "An organisation must not collect, use or disclose personal data unless...\n"
            "24 Protection of personal data\n"
            "An organisation must protect personal data in its possession or under its control.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 13", "Section 24"]
    assert units[0].canonical_citation == "Personal Data Protection Act 2012 Section 13"
    assert "Consent required" in units[0].text


def test_commonwealth_parser_extracts_hong_kong_section_heading():
    doc = SourceDocument(
        doc_id="hong_kong_pdpo_cap486",
        jurisdiction="HK",
        law_family="hong_kong_pdpo",
        source_type="primary_law",
        title="Personal Data (Privacy) Ordinance Cap. 486",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Part 1 Preliminary\n"
            "Section: 2 Interpretation E.R. 1 of 2013 25/04/2013\n"
            "In this Ordinance, unless the context otherwise requires...\n"
            "Section: 4 Data protection principles E.R. 1 of 2013 25/04/2013\n"
            "A data user shall not do an act that contravenes a data protection principle.\n"
            "1. Principle 1-purpose and manner of collection of personal data\n"
            "Personal data shall not be collected unless...\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 2", "Section 4"]
    assert units[0].canonical_citation == "Personal Data (Privacy) Ordinance Cap. 486 Section 2"
    assert "Interpretation" in units[0].text


def test_commonwealth_parser_extracts_hong_kong_schedule_one_principles_separately():
    doc = SourceDocument(
        doc_id="hong_kong_pdpo_cap486",
        jurisdiction="HK",
        law_family="hong_kong_pdpo",
        source_type="primary_law",
        title="Personal Data (Privacy) Ordinance Cap. 486",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Section: 72 (Omitted as spent—E.R. 1 of 2013) E.R. 1 of 2013 25/04/2013\n"
            "Section 72 body.\n"
            "Section: 73 (Omitted as spent—E.R. 1 of 2013) E.R. 1 of 2013 25/04/2013\n"
            "Section 73 body.\n"
            "Schedule: 1 Data Protection Principles E.R. 1 of 2013 25/04/2013\n"
            "[sections 2(1) & (6)]\n"
            "1. Principle 1-purpose and manner of collection of personal data\n"
            "Personal data shall not be collected unless.\n"
            "2. Principle 2-accuracy and duration of retention of personal data\n"
            "Personal data shall be accurate.\n"
            "Schedule: 2 Classes of Data Users E.R. 1 of 2013 25/04/2013\n"
            "1. A schedule table entry.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 72",
        "Section 73",
        "Schedule 1 Principle 1",
        "Schedule 1 Principle 2",
        "Schedule 2",
    ]
    assert "Principle 1" not in units[1].text
    assert units[2].unit_type == "schedule_clause"


def test_commonwealth_parser_filters_india_page_headers():
    doc = SourceDocument(
        doc_id="india_dpdp_act_2023",
        jurisdiction="IN",
        law_family="india_dpdp",
        source_type="primary_law",
        title="Digital Personal Data Protection Act, 2023",
        version_date="2023-08-11",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "CHAPTER I\n"
            "PRELIMINARY\n"
            "1. (1) This Act may be called the Digital Personal Data Protection Act, 2023.\n"
            "(2) It shall come into force on such date as the Central Government may notify.\n"
            "2 THE GAZETTE OF INDIA EXTRAORDINARY [PART II-\n"
            "2. In this Act, unless the context otherwise requires,-\n"
            "(a) Appellate Tribunal means the Telecom Disputes Settlement and Appellate Tribunal.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 1", "Section 2"]
    assert "THE GAZETTE" not in units[0].local_citation
    assert "Digital Personal Data Protection Act" in units[0].text


def test_commonwealth_parser_extracts_india_penalty_schedule_items_separately():
    doc = SourceDocument(
        doc_id="india_dpdp_act_2023",
        jurisdiction="IN",
        law_family="india_dpdp",
        source_type="primary_law",
        title="Digital Personal Data Protection Act, 2023",
        version_date="2023-08-11",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "CHAPTER I\n"
            "PRELIMINARY\n"
            "1. (1) This Act may be called the Digital Personal Data Protection Act, 2023.\n"
            "44. (1) The enactments specified in the Schedule are hereby amended.\n"
            "Breach of provisions of  this Act or rules made thereunder\n"
            "(2)\n"
            "Breach in observing the obligation of Data Fiduciary to take reasonable security "
            "safeguards to prevent personal data breach under sub-section (5) of section 8.\n"
            "Breach in observing the obligation to give the Board or affected Data Principal "
            "notice of a personal data breach under sub-section (6) of section 8.\n"
            "Breach in observance of additional obligations in relation to children under section 9.\n"
            "Breach in observance of additional obligations of Significant Data Fiduciary under "
            "section 10.\n"
            "Breach in observance of the duties under section 15.\n"
            "Breach of any term of voluntary undertaking accepted by the Board under section 32.\n"
            "Breach of any other provision of this Act or the rules made thereunder.\n"
            "Sl. No.\n"
            "(1)\n"
            "1.\n"
            "2.\n"
            "3.\n"
            "4.\n"
            "5.\n"
            "6.\n"
            "7.\n"
            "Penalty\n"
            "(3)\n"
            "May extend to two hundred and fifty crore rupees.\n"
            "May extend to two hundred crore rupees.\n"
            "May extend to two hundred crore rupees.\n"
            "May extend to one hundred and fifty crore rupees.\n"
            "May extend to ten thousand rupees.\n"
            "Up to the extent applicable for the breach in respect of which the proceedings "
            "under section 28 were instituted.\n"
            "May extend to fifty crore rupees.\n"
            "THE  SCHEDULE\n"
            "[See section 33 (1)]\n"
            "DR. REETA VASISHTA,\n"
            "Secy. to the Govt. of India.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 1",
        "Section 44",
        "Schedule item 1",
        "Schedule item 2",
        "Schedule item 3",
        "Schedule item 4",
        "Schedule item 5",
        "Schedule item 6",
        "Schedule item 7",
    ]
    schedule_items = units[2:]
    assert [unit.unit_type for unit in schedule_items] == ["schedule_clause"] * 7
    assert units[2].canonical_citation == (
        "Digital Personal Data Protection Act, 2023 Schedule item 1"
    )
    assert "reasonable security safeguards" in units[2].text
    assert "two hundred and fifty crore rupees" in units[2].text
    assert "notice of a personal data breach" in units[3].text
    assert "two hundred crore rupees" in units[3].text
    assert "voluntary undertaking accepted by the Board" in units[7].text
    assert "Up to the extent applicable" in units[7].text
    assert "any other provision of this Act" in units[8].text
    assert "fifty crore rupees" in units[8].text
    assert all("DR. REETA" not in unit.text for unit in schedule_items)
    assert "Breach of provisions" not in units[1].text


def test_commonwealth_parser_ignores_singapore_legislative_history_duplicates():
    doc = SourceDocument(
        doc_id="singapore_pdpa_2012",
        jurisdiction="SG",
        law_family="singapore_pdpa",
        source_type="primary_law",
        title="Personal Data Protection Act 2012",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "Part 1 PRELIMINARY\n"
            "1 Short title\n"
            "This Act is the Personal Data Protection Act 2012.\n"
            "2 Interpretation\n"
            "In this Act, unless the context otherwise requires...\n"
            "Legislative History\n"
            "1 Jan 2021\n"
            "2020 REVISED EDITION\n"
            "1 Short title\n"
            "This duplicate should not be parsed.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 1", "Section 2"]
    assert "duplicate" not in units[-1].text


def test_commonwealth_parser_keeps_singapore_schedules_out_of_sections():
    doc = SourceDocument(
        doc_id="singapore_pdpa_2012",
        jurisdiction="SG",
        law_family="singapore_pdpa",
        source_type="primary_law",
        title="Personal Data Protection Act 2012",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "67 Saving and transitional provisions\n"
            "Section 67 body.\n"
            "68 Dissolution\n"
            "Section 68 body.\n"
            "FIRST SCHEDULE\n"
            "COLLECTION, USE AND DISCLOSURE OF PERSONAL DATA WITHOUT CONSENT\n"
            "1. Schedule paragraph one.\n"
            "FIRST SCHEDULE — continued\n"
            "2A. Continued schedule paragraph should not be a section.\n"
            "SECOND SCHEDULE\n"
            "ADDITIONAL BASES\n"
            "1. Second schedule paragraph.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 67",
        "Section 68",
        "First Schedule",
        "Second Schedule",
    ]
    assert [unit.unit_type for unit in units[-2:]] == ["schedule_clause", "schedule_clause"]
    assert all(unit.local_citation != "Section 2A" for unit in units)


def test_commonwealth_parser_extracts_canada_number_lines_and_schedule_one():
    doc = SourceDocument(
        doc_id="canada_pipeda_current",
        jurisdiction="CA",
        law_family="canada_pipeda",
        source_type="primary_law",
        title="Personal Information Protection and Electronic Documents Act",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/canada-pipeda",
        language="en",
        raw_text=(
            "Table of Contents\n"
            "1 - Short Title\n"
            "2 - PART 1\n"
            "Short Title\n"
            "1\n"
            "This Act may be cited as the Personal Information Protection and Electronic Documents Act.\n"
            "2\n"
            "(1) The definitions in this subsection apply in this Part.\n"
            "3\n"
            "The purpose of this Part is to establish rules to govern personal information.\n"
            "SCHEDULE 1\n"
            "4.1 Principle 1 — Accountability\n"
            "An organization is responsible for personal information under its control.\n"
            "4.1.1\n"
            "Accountability rests with the designated individual.\n"
            "4.10.4\n"
            "An organization shall investigate all complaints.\n"
            "SCHEDULE 2\n"
            "1\n"
            "This schedule table should not be parsed.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 1",
        "Section 2",
        "Section 3",
        "Schedule 1 clause 4.1",
        "Schedule 1 clause 4.1.1",
        "Schedule 1 clause 4.10.4",
    ]
    assert "Table of Contents" not in units[0].text
    assert "schedule table" not in units[-1].text
    assert [unit.unit_type for unit in units[-3:]] == [
        "schedule_clause",
        "schedule_clause",
        "schedule_clause",
    ]


def test_commonwealth_parser_uses_malaysia_english_body_not_bilingual_toc():
    doc = SourceDocument(
        doc_id="malaysia_pdpa_2010",
        jurisdiction="MY",
        law_family="malaysia_pdpa",
        source_type="primary_law",
        title="Personal Data Protection Act 2010",
        version_date="2010",
        effective_date=None,
        source_url="https://example.test/malaysia-pdpa",
        language="en",
        raw_text=(
            "SUSUNAN SEKSYEN\n"
            "1. Tajuk ringkas dan permulaan kuat kuasa\n"
            "2. Pemakaian\n"
            "ARRANGEMENT OF SECTIONS\n"
            "1. Short title and commencement\n"
            "2. Application\n"
            "An Act to regulate the processing of personal data in commercial\n"
            "transactions and to provide for related matters.\n"
            "Short title and commencement\n"
            "1. (1) This Act may be cited as the Personal Data Protection Act 2010.\n"
            "A reference to section 2. does not start a new legal unit.\n"
            "Application\n"
            "2. (1) This Act applies to a person who processes personal data.\n"
            "Non-application\n"
            "3. This Act does not apply to the Federal Government.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 1",
        "Section 2",
        "Section 3",
    ]
    assert units[0].text.startswith("1. (1) This Act may be cited")
    assert "Tajuk ringkas" not in units[0].text
    assert "reference to section 2" in units[0].text
    assert units[-1].text.endswith("Federal Government.")


def test_commonwealth_parser_extracts_australia_sections_and_app_clauses_separately():
    doc = SourceDocument(
        doc_id="australia_privacy_act_1988_current",
        jurisdiction="AU",
        law_family="australia_privacy_act",
        source_type="primary_law",
        title="Privacy Act 1988",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/australia-privacy-act",
        language="en",
        raw_text=(
            "Table of contents\n"
            "1\n"
            "Short title\n"
            "Schedule\n"
            "1\n"
            "—\n"
            "Australian Privacy Principles\n"
            "Endnotes\n"
            "Endnote 1—About the endnotes\n"
            "BE IT THEREFORE ENACTED by the Queen as follows:\n"
            "Part\n"
            "I\n"
            "—\n"
            "Preliminary\n"
            "1\n"
            "Short title\n"
            "This Act may be cited as the Privacy Act 1988.\n"
            "2A\n"
            "Objects of this Act\n"
            "The objects of this Act are to promote privacy.\n"
            "16A\n"
            "Permitted general situations\n"
            "The following table applies.\n"
            "5\n"
            "APP entity\n"
            "This table item should not be parsed as section 5.\n"
            "Schedule\n"
            "1\n"
            "—\n"
            "Australian Privacy Principles\n"
            "Overview of the Australian Privacy Principles\n"
            "Part\n"
            "1\n"
            "—\n"
            "Consideration of personal information privacy\n"
            "1\n"
            "Australian Privacy Principle\n"
            "1—open and transparent management of personal information\n"
            "1.1\n"
            "The object of this principle is to ensure APP entities manage personal information openly.\n"
            "13\n"
            "Australian Privacy Principle\n"
            "13—correction of personal information\n"
            "13.5\n"
            "If a request is made, the APP entity must respond.\n"
            "Schedule\n"
            "2\n"
            "—\n"
            "Statutory Tort for Serious Invasions of Privacy\n"
            "1\n"
            "Objects of this Schedule\n"
            "This schedule should not be parsed as a Privacy Act section.\n"
            "Endnotes\n"
            "Endnote 1—About the endnotes\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 1",
        "Section 2A",
        "Section 16A",
        "APP 1",
        "APP 1.1",
        "APP 13",
        "APP 13.5",
        "Schedule 2 clause 1",
    ]
    assert units[3].unit_type == "schedule_clause"
    assert "Table of contents" not in units[0].text
    assert units[-1].unit_type == "schedule_clause"
    assert "Objects of this Schedule" in units[-1].text
    assert all(unit.local_citation != "Section 5" for unit in units)


def test_commonwealth_parser_extracts_uk_schedule_paragraphs_separately():
    doc = SourceDocument(
        doc_id="uk_data_protection_act_2018",
        jurisdiction="UK",
        law_family="uk_dpa",
        source_type="primary_law",
        title="Data Protection Act 2018",
        version_date="current",
        effective_date=None,
        source_url="https://www.legislation.gov.uk/ukpga/2018/12/data.xml",
        language="en",
        raw_text=(
            "1\n"
            "Overview\n"
            "This Act makes provision about the processing of personal data.\n"
            "2\n"
            "Protection of personal data\n"
            "The UK GDPR and this Act protect individuals.\n\n"
            "SCHEDULE A1\n"
            "Processing in reliance on relevant international law\n"
            "This condition is met for a request made under the agreement.\n\n"
            "SCHEDULE 2\n"
            "Exemptions etc from the UK GDPR\n"
            "Paragraph 1\n"
            "The listed GDPR provisions are defined for this Part.\n"
            "Paragraph 2\n"
            "The listed GDPR provisions do not apply to personal data processed for crime.\n"
            "SCHEDULE 12A\n"
            "The Information Commission\n"
            "Paragraph 1\n"
            "The Commission is a body corporate.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 1",
        "Section 2",
        "Schedule A1",
        "Schedule 2 paragraph 1",
        "Schedule 2 paragraph 2",
        "Schedule 12A paragraph 1",
    ]
    assert [unit.unit_type for unit in units] == [
        "section",
        "section",
        "schedule_clause",
        "schedule_clause",
        "schedule_clause",
        "schedule_clause",
    ]
    assert "Exemptions etc" in units[3].canonical_citation
    assert "crime" in units[4].text
    assert "Information Commission" in units[-1].canonical_citation


def test_commonwealth_parser_uses_uk_xml_boundaries_and_revision_status():
    doc = SourceDocument(
        doc_id="uk_data_protection_act_2018",
        jurisdiction="UK",
        law_family="uk_dpa",
        source_type="primary_law",
        title="Data Protection Act 2018",
        version_date="current",
        effective_date="2018-05-25",
        source_url="https://www.legislation.gov.uk/ukpga/2018/12/data.xml",
        language="en",
        raw_text=(
            "20\n"
            "Meaning of “court”\n"
            ". . . . . . . .\n\n"
            "114A\n"
            "The Information Commission\n"
            "1 A body corporate called the Information Commission is established. "
            "2 Schedule 12A makes further provision about the Commission.\n\n"
            "115\n"
            "General functions\n"
            "The Commission has the following functions."
        ),
        metadata={
            "uk_legislation": {
                "schema_version": "1.0",
                "sections": {
                    "20": {
                        "title": "Meaning of “court”",
                        "status": "omitted",
                        "effective_from": None,
                        "effective_to": "2025-08-20",
                    },
                    "114A": {
                        "title": "The Information Commission",
                        "status": "active",
                        "effective_from": "2025-08-20",
                        "effective_to": None,
                    },
                    "115": {
                        "title": "General functions",
                        "status": "active",
                        "effective_from": None,
                        "effective_to": None,
                    },
                },
            }
        },
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 20",
        "Section 114A",
        "Section 115",
    ]
    assert units[0].is_current is False
    assert units[0].effective_to == "2025-08-20"
    assert units[1].effective_from == "2025-08-20"
    assert "body corporate" in units[1].text
    assert "Schedule 12A" in units[1].text


def test_commonwealth_parser_handles_singapore_pdf_title_number_layout():
    doc = SourceDocument(
        doc_id="singapore_pdpa_2012",
        jurisdiction="SG",
        law_family="singapore_pdpa",
        source_type="primary_law",
        title="Personal Data Protection Act 2012",
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=(
            "P AR T 1\n"
            "PRELIMINAR Y\n"
            "Short title\n"
            "1. This Act is the Personal Data Protection Act 2012.\n"
            "Interpre tation\n"
            "2.—(1) In this Act, unless the context otherwise requires —\n"
            "Compliance with Act\n"
            "1 1. In meeting its responsibilities under this Act, an organisation shall consider what a reasonable person would consider appropriate.\n"
            "P AR T 7\n"
            "27. [Repealed]\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 1", "Section 2", "Section 11"]
    assert "reasonable person" in units[-1].text


def test_commonwealth_parser_extracts_major_market_numbered_sections():
    doc = SourceDocument(
        doc_id="uk_data_protection_act_2018",
        jurisdiction="UK",
        law_family="uk_dpa",
        source_type="primary_law",
        title="Data Protection Act 2018",
        version_date="current",
        effective_date=None,
        source_url="https://www.legislation.gov.uk/ukpga/2018/12/data.xml",
        language="en",
        raw_text=(
            "PART 1\n"
            "PRELIMINARY\n"
            "1\n"
            "Overview\n"
            "This Act makes provision about the processing of personal data.\n"
            "2 Protection of personal data\n"
            "The GDPR, the applied GDPR and this Act protect individuals.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 1", "Section 2"]
    assert units[0].canonical_citation == "Data Protection Act 2018 Section 1"
    assert "Overview" in units[0].text
    assert "Protection of personal data" in units[1].text


@pytest.mark.parametrize(
    (
        "doc_id",
        "jurisdiction",
        "law_family",
        "title",
        "raw_text",
        "expected_citations",
    ),
    [
        (
            "canada_pipeda_current",
            "CA",
            "canada_pipeda",
            "Personal Information Protection and Electronic Documents Act",
            (
                "PART 1\n"
                "PROTECTION OF PERSONAL INFORMATION IN THE PRIVATE SECTOR\n"
                "2 Definitions\n"
                "In this Part, personal information means information about an identifiable individual.\n"
                "5 Compliance with obligations\n"
                "Every organization shall comply with the obligations set out in Schedule 1.\n"
            ),
            ["Section 2", "Section 5"],
        ),
        (
            "australia_privacy_act_1988_current",
            "AU",
            "australia_privacy_act",
            "Privacy Act 1988",
            (
                "Part I Preliminary\n"
                "6 Interpretation\n"
                "In this Act, personal information means information or an opinion about an identified individual.\n"
                "13 Interferences with privacy\n"
                "An act or practice is an interference with the privacy of an individual if...\n"
            ),
            ["Section 6", "Section 13"],
        ),
        (
            "south_africa_popia_act_2013",
            "ZA",
            "south_africa_popia",
            "Protection of Personal Information Act, 2013",
            (
                "CHAPTER 1\n"
                "DEFINITIONS AND PURPOSE\n"
                "1 Definitions\n"
                "In this Act, unless the context indicates otherwise, biometric means a technique.\n"
                "2 Purpose of Act\n"
                "The purpose of this Act is to give effect to the constitutional right to privacy.\n"
            ),
            ["Section 1", "Section 2"],
        ),
        (
            "philippines_data_privacy_act_2012",
            "PH",
            "philippines_dpa",
            "Data Privacy Act of 2012",
            (
                "CHAPTER I GENERAL PROVISIONS\n"
                "4 Scope\n"
                "This Act applies to the processing of all types of personal information.\n"
                "11 General Data Privacy Principles\n"
                "The processing of personal information shall be allowed subject to compliance.\n"
            ),
            ["Section 4", "Section 11"],
        ),
        (
            "malaysia_pdpa_2010",
            "MY",
            "malaysia_pdpa",
            "Personal Data Protection Act 2010",
            (
                "PART I PRELIMINARY\n"
                "4 Application\n"
                "This Act applies to any person who processes personal data.\n"
                "5 Personal data protection principles\n"
                "A data user shall comply with the Personal Data Protection Principles.\n"
            ),
            ["Section 4", "Section 5"],
        ),
    ],
)
def test_commonwealth_parser_extracts_registered_numbered_family_samples(
    doc_id,
    jurisdiction,
    law_family,
    title,
    raw_text,
    expected_citations,
):
    doc = SourceDocument(
        doc_id=doc_id,
        jurisdiction=jurisdiction,
        law_family=law_family,
        source_type="primary_law",
        title=title,
        version_date="current",
        effective_date=None,
        source_url="https://example.test",
        language="en",
        raw_text=raw_text,
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == expected_citations
    assert all(unit.unit_type == "section" for unit in units)
    assert all(unit.text for unit in units)


def test_commonwealth_parser_extracts_japan_article_headings():
    doc = SourceDocument(
        doc_id="japan_appi",
        jurisdiction="JP",
        law_family="japan_appi",
        source_type="primary_law",
        title="Act on the Protection of Personal Information",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/japan-appi",
        language="en",
        raw_text=(
            "Chapter I General Provisions\n"
            "Article 1 (Purpose)\n"
            "The purpose of this Act is to protect the rights and interests of individuals.\n"
            "Article 2 (Definitions)\n"
            "In this Act, personal information means information about a living individual.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 1", "Article 2"]
    assert [unit.unit_type for unit in units] == ["article", "article"]
    assert units[0].canonical_citation == "Act on the Protection of Personal Information Article 1"
    assert "rights and interests" in units[0].text
    assert units[0].span_ids[0].startswith("japan_appi:span:")


def test_commonwealth_parser_keeps_japan_body_articles_over_toc_duplicates():
    doc = SourceDocument(
        doc_id="japan_appi",
        jurisdiction="JP",
        law_family="japan_appi",
        source_type="primary_law",
        title="Act on the Protection of Personal Information",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/japan-appi",
        language="en",
        raw_text=(
            "Table of Contents\n"
            "Article 1 (Purpose)\n"
            "Article 2 (Definitions)\n"
            "Chapter I General Provisions\n"
            "Article 1\n"
            "Purpose\n"
            "The purpose of this Act is to protect the rights and interests of individuals.\n"
            "Article 2\n"
            "Definitions\n"
            "In this Act, personal information means information about a living individual.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 1", "Article 2"]
    assert units[0].text.startswith("Article 1\nPurpose")
    assert "rights and interests" in units[0].text
    assert "living individual" in units[1].text
    assert "Chapter I General Provisions" not in units[0].text


def test_commonwealth_parser_ignores_japan_web_toc_before_body_articles():
    doc = SourceDocument(
        doc_id="japan_appi",
        jurisdiction="JP",
        law_family="japan_appi",
        source_type="primary_law",
        title="Act on the Protection of Personal Information",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/japan-appi",
        language="en",
        raw_text=(
            "Table of Contents\n"
            "Article 108\n"
            "Section 5 Provision of Anonymized Personal Information Administrative Entities Hold.\n"
            "Article 185\n"
            "Appended Table 1\n"
            "Article 1 (Purpose)\n"
            "Article 2 (Definitions)\n"
            "Act on the Protection of Personal Information\n"
            "Article 1\n"
            "The purpose of this Act is to protect the rights and interests of individuals.\n"
            "Article 2\n"
            "In this Act, personal information means information about a living individual.\n"
            "Article 108\n"
            "The provisions in this Section do not preclude a local government from acting.\n"
            "Article 185\n"
            "A person falling under one of the following items is subject to a civil fine.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Article 1",
        "Article 2",
        "Article 108",
        "Article 185",
    ]
    assert units[0].text.startswith("Article 1\nThe purpose")
    assert "Section 5 Provision" not in units[2].text


def test_commonwealth_parser_extracts_korea_article_headings():
    doc = SourceDocument(
        doc_id="korea_pipa",
        jurisdiction="KR",
        law_family="korea_pipa",
        source_type="primary_law",
        title="Personal Information Protection Act",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/korea-pipa",
        language="en",
        raw_text=(
            "CHAPTER I GENERAL PROVISIONS\n"
            "Article 1\n"
            "Purpose\n"
            "The purpose of this Act is to prescribe matters concerning processing.\n"
            "Article 2\n"
            "Definitions\n"
            "The terms used in this Act shall be defined as follows.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 1", "Article 2"]
    assert units[0].unit_type == "article"
    assert units[1].canonical_citation == "Personal Information Protection Act Article 2"
    assert "defined as follows" in units[1].text


def test_commonwealth_parser_keeps_first_meaningful_article_duplicate():
    doc = SourceDocument(
        doc_id="korea_pipa",
        jurisdiction="KR",
        law_family="korea_pipa",
        source_type="primary_law",
        title="Personal Information Protection Act",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/korea-pipa",
        language="en",
        raw_text=(
            "CHAPTER I GENERAL PROVISIONS\n"
            "Article 1 (Purpose)\n"
            "The purpose of this Act is to protect personal information.\n"
            "Article 2 (Definitions)\n"
            "The terms used in this Act are defined as follows.\n"
            "ADDENDA\n"
            "Article 1 (Enforcement Date)\n"
            "This Act shall enter into force on the date of its promulgation.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 1", "Article 2"]
    assert "protect personal information" in units[0].text
    assert "Enforcement Date" not in units[0].text


def test_commonwealth_parser_extracts_hyphenated_and_deleted_articles():
    doc = SourceDocument(
        doc_id="korea_pipa",
        jurisdiction="KR",
        law_family="korea_pipa",
        source_type="primary_law",
        title="Personal Information Protection Act",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/korea-pipa",
        language="en",
        raw_text=(
            "Article 7-2 (Composition of the Protection Commission)\n"
            "The Protection Commission shall be composed of members.\n"
            "Article 8 Deleted. <Feb. 4, 2020>\n"
            "Article 8-2 (Assessment of Personal Information Breach Incident Factors)\n"
            "The head of a central administrative agency shall request an assessment.\n"
            "ADDENDA\n"
            "Article 8 (Transitional Measures)\n"
            "The previous provisions shall apply.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Article 7-2",
        "Article 8",
        "Article 8-2",
    ]
    assert "Deleted" in units[1].text
    assert "Transitional Measures" not in units[1].text


def test_commonwealth_parser_extracts_philippines_section_word_headings():
    doc = SourceDocument(
        doc_id="philippines_data_privacy_act_2012",
        jurisdiction="PH",
        law_family="philippines_dpa",
        source_type="primary_law",
        title="Data Privacy Act of 2012",
        version_date="2012",
        effective_date=None,
        source_url="https://example.test/philippines-dpa",
        language="en",
        raw_text=(
            "CHAPTER I\n"
            "GENERAL PROVISIONS\n"
            "Section 4.\n"
            "Scope.\n"
            "This Act applies to the processing of all types of personal information.\n"
            "Section 11.\n"
            "General Data Privacy Principles.\n"
            "The processing of personal information shall be allowed subject to compliance.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Section 4", "Section 11"]
    assert "all types of personal information" in units[0].text
    assert "subject to compliance" in units[1].text


def test_commonwealth_parser_uses_south_africa_body_not_contents_or_schedule():
    doc = SourceDocument(
        doc_id="south_africa_popia_act_2013",
        jurisdiction="ZA",
        law_family="south_africa_popia",
        source_type="primary_law",
        title="Protection of Personal Information Act, 2013",
        version_date="2013",
        effective_date=None,
        source_url="https://example.test/south-africa-popia",
        language="en",
        raw_text=(
            "CONTENTS OF ACT\n"
            "CHAPTER 1\n"
            "DEFINITIONS AND PURPOSE\n"
            "1. Definitions\n"
            "2. Purpose of Act\n"
            "12 No. 37067 GOVERNMENT GAZETTE, 26 November 2013\n"
            "Act No. 4 of 2013 Protection of Personal Information Act, 2013\n"
            "SCHEDULE\n"
            "Laws amended by section 110\n"
            "CHAPTER 1\n"
            "DEFINITIONS AND PURPOSE\n"
            "Definitions\n"
            "1. In this Act, unless the context indicates otherwise, biometric means a technique.\n"
            "Purpose of Act\n"
            "2. The purpose of this Act is to give effect to the constitutional right to privacy.\n"
            "Application and interpretation of Act\n"
            "3. (1) This Act applies to the processing of personal information.\n"
            "Lawfulness of processing\n"
            "9. Personal information must be processed lawfully.\n"
            "14 No. 37067 GOVERNMENT GAZETTE, 26 November 2013\n"
            "Amendment of laws\n"
            "110.The laws mentioned in the Schedule are amended.\n"
            "Short title and commencement\n"
            "115. This Act is called the Protection of Personal Information Act, 2013.\n"
            "SCHEDULE\n"
            "LAWS AMENDED BY SECTION 110\n"
            "77K. This schedule amendment should not be parsed as a POPIA section.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Section 1",
        "Section 2",
        "Section 3",
        "Section 9",
        "Section 110",
        "Section 115",
    ]
    assert "Definitions" not in units[0].local_citation
    assert "GOVERNMENT GAZETTE" not in "\n".join(unit.text for unit in units)
    assert "77K" not in [unit.local_citation for unit in units]


def test_commonwealth_parser_extracts_brazil_art_headings():
    doc = SourceDocument(
        doc_id="brazil_lgpd",
        jurisdiction="BR",
        law_family="brazil_lgpd",
        source_type="primary_law",
        title="Lei Geral de Protecao de Dados Pessoais",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/brazil-lgpd",
        language="en",
        raw_text=(
            "CHAPTER I\n"
            "PRELIMINARY PROVISIONS\n"
            "Art. 1\n"
            "This Law provides for the processing of personal data.\n"
            "Art. 2\n"
            "The discipline of personal data protection is based on respect for privacy.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 1", "Article 2"]
    assert units[0].canonical_citation == "Lei Geral de Protecao de Dados Pessoais Article 1"
    assert "respect for privacy" in units[1].text


def test_commonwealth_parser_extracts_brazil_same_line_article_starts():
    doc = SourceDocument(
        doc_id="brazil_lgpd",
        jurisdiction="BR",
        law_family="brazil_lgpd",
        source_type="primary_law",
        title="Lei Geral de Protecao de Dados Pessoais",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/brazil-lgpd",
        language="en",
        raw_text=(
            "CHAPTER I\n"
            "PRELIMINARY PROVISIONS\n"
            "Article 1 of another law is mentioned in this preface.\n"
            "Article 1. This Law provides for the processing of personal data.\n"
            "References to Article 1. inside a sentence must stay in the same unit.\n"
            "Article 2. The discipline of personal data protection is based on respect for privacy.\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["Article 1", "Article 2"]
    assert "processing of personal data" in units[0].text
    assert "inside a sentence" in units[0].text
    assert units[1].text.startswith("Article 2. The discipline")


def test_commonwealth_parser_extracts_brazil_ocr_article_heading_variants():
    doc = SourceDocument(
        doc_id="brazil_lgpd",
        jurisdiction="BR",
        law_family="brazil_lgpd",
        source_type="primary_law",
        title="Lei Geral de Protecao de Dados Pessoais",
        version_date="current",
        effective_date=None,
        source_url="https://example.test/brazil-lgpd",
        language="en",
        raw_text=(
            "Article 6. Activities of processing shall follow principles.\n"
            "Article 7 . Processing of personal data shall only be carried out under conditions.\n"
            "Article 8. The consent set forth in item I of article 7 of this Law shall be written.\n"
            ". Article 17. Every natural person is assured ownership of their personal data.\n"
            "Article 18. Data subjects shall have rights.\n"
            "Article 55. (vetoed)\n"
            "Article 55 -A. The National Data Protection Authority is hereby created.\n"
            "Article 55-B. (Repealed by Law No. 14,460/2022)\n"
            "Article 60. Law No. 12,965 is amended as follows:\n"
            "“Article 7. (...)\n"
            "X - definitive exclusion of personal data.”\n"
        ),
    )

    units = CommonwealthLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == [
        "Article 6",
        "Article 7",
        "Article 8",
        "Article 17",
        "Article 18",
        "Article 55",
        "Article 55-A",
        "Article 55-B",
        "Article 60",
    ]
    by_citation = {unit.local_citation: unit for unit in units}
    assert by_citation["Article 6"].text == (
        "Article 6. Activities of processing shall follow principles."
    )
    assert by_citation["Article 7"].unit_id == "brazil_lgpd:article_7"
    assert by_citation["Article 55-A"].unit_id == "brazil_lgpd:article_55-a"
    assert "“Article 7" in by_citation["Article 60"].text
