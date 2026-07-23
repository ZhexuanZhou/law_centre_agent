from pathlib import Path

from crawler.law_corpus.catalog import load_sources
from crawler.law_corpus.parsers.registry import get_parser


def test_load_sources_from_toml(tmp_path: Path):
    catalog = tmp_path / "laws.seed.toml"
    catalog.write_text(
        """
[[sources]]
doc_id = "eu_gdpr_2016_679"
title = "GDPR"
jurisdiction = "EU"
law_family = "eu_gdpr"
source_type = "primary_law"
version_date = "2016-04-27"
effective_date = "2018-05-25"
language = "en"
url = "https://example.test/gdpr"
preferred_format = "html"
download_mode = "auto"
target_path = "data/raw/laws/EU/eu_gdpr/2016-04-27/eu_gdpr_2016_679.html"
""",
        encoding="utf-8",
    )

    sources = load_sources(catalog)

    assert len(sources) == 1
    assert sources[0].doc_id == "eu_gdpr_2016_679"
    assert sources[0].requires_manual_fetch is False


def test_seed_catalog_uses_ecfr_api_xml_for_us_cfr_sources():
    sources = load_sources("corpus/sources/laws.seed.toml")
    us_cfr_sources = {source.doc_id: source for source in sources if source.law_family == "us_cfr"}

    assert set(us_cfr_sources) == {
        "us_hipaa_45_cfr_part_164",
        "us_coppa_16_cfr_part_312",
        "us_glba_16_cfr_part_314",
    }
    for source in us_cfr_sources.values():
        assert source.url.startswith("https://www.ecfr.gov/api/versioner/v1/full/")
        assert source.url.endswith(".xml?part=" + source.doc_id.rsplit("_", 1)[-1])
        assert source.preferred_format == "xml"
        assert source.target_path.endswith(".xml")


def test_load_sources_defaults_package_metadata_for_legacy_catalog(tmp_path: Path):
    catalog = tmp_path / "laws.seed.toml"
    catalog.write_text(
        """
[[sources]]
doc_id = "eu_gdpr_2016_679"
title = "GDPR"
jurisdiction = "EU"
law_family = "eu_gdpr"
source_type = "primary_law"
version_date = "2016-04-27"
effective_date = "2018-05-25"
language = "en"
url = "https://example.test/gdpr"
preferred_format = "html"
download_mode = "auto"
target_path = "corpus/raw/laws/EU/eu_gdpr/2016-04-27/eu_gdpr_2016_679.html"
""",
        encoding="utf-8",
    )

    source = load_sources(catalog)[0]

    assert source.source_set == "seed"
    assert source.translation_status == "official_original"


def test_load_sources_reads_expansion_package_metadata(tmp_path: Path):
    catalog = tmp_path / "laws.expansion_major_markets.toml"
    catalog.write_text(
        """
[[sources]]
doc_id = "uk_data_protection_act_2018"
title = "Data Protection Act 2018"
jurisdiction = "UK"
law_family = "uk_dpa"
source_type = "primary_law"
version_date = "current"
effective_date = ""
language = "en"
url = "https://www.legislation.gov.uk/ukpga/2018/12/data.xml"
preferred_format = "xml"
download_mode = "auto"
target_path = "corpus/raw/laws/UK/uk_dpa/current/uk_data_protection_act_2018.xml"
source_set = "expansion_major_markets"
translation_status = "official_original"
""",
        encoding="utf-8",
    )

    source = load_sources(catalog)[0]

    assert source.source_set == "expansion_major_markets"
    assert source.translation_status == "official_original"


def test_expansion_catalogs_have_source_sets_and_unique_doc_ids():
    expected_source_sets = {
        "corpus/sources/laws.seed.toml": "seed",
        "corpus/sources/laws.expansion_major_markets.toml": "expansion_major_markets",
        "corpus/sources/laws.expansion_eu_data_rules.toml": "expansion_eu_data_rules",
    }
    allowed_translation_statuses = {
        "official_original",
        "official_translation",
        "authoritative_translation",
        "unofficial_translation",
    }
    extension_by_format = {
        "html": ".html",
        "xml": ".xml",
        "pdf": ".pdf",
        "txt": ".txt",
    }
    expected_expansion_doc_ids = {
        "brazil_lgpd_law_13709_en",
        "china_cybersecurity_law_2016",
        "eu_data_act_2023_2854",
        "eu_data_governance_act_2022_868",
        "eu_digital_services_act_2022_2065",
        "eu_nis2_directive_2022_2555",
        "japan_appi_current_en",
        "korea_pipa_current_en",
        "malaysia_pdpa_2010",
        "philippines_data_privacy_act_2012",
        "south_africa_popia_act_2013",
        "uk_data_protection_act_2018",
        "us_ca_ccpa_cpra_civ_1798_100_199",
    }
    all_doc_ids: list[str] = []
    manual_doc_ids: set[str] = set()
    sources_by_doc_id = {}
    for catalog_path, expected_source_set in expected_source_sets.items():
        sources = load_sources(catalog_path)
        assert sources, catalog_path
        for source in sources:
            sources_by_doc_id[source.doc_id] = source
            assert source.source_set == expected_source_set
            assert source.translation_status in allowed_translation_statuses
            assert source.target_path.startswith("corpus/raw/laws/")
            assert get_parser(source.law_family)

            if source.requires_manual_fetch:
                manual_doc_ids.add(source.doc_id)
                assert source.manual_instructions.strip()
                assert "target_path" in source.manual_instructions

            assert source.preferred_format in extension_by_format
            assert source.target_path.endswith(extension_by_format[source.preferred_format])

            all_doc_ids.append(source.doc_id)

    assert len(all_doc_ids) == len(set(all_doc_ids))
    actual_doc_ids = set(all_doc_ids)
    assert expected_expansion_doc_ids <= actual_doc_ids
    assert not (expected_expansion_doc_ids & manual_doc_ids)

    eu_html_doc_ids = {
        "eu_data_act_2023_2854",
        "eu_data_governance_act_2022_868",
        "eu_digital_services_act_2022_2065",
        "eu_nis2_directive_2022_2555",
    }
    for doc_id in eu_html_doc_ids:
        source = sources_by_doc_id[doc_id]
        assert source.download_mode == "auto"
        assert source.url.startswith("https://eur-lex.europa.eu/eli/")
        assert source.preferred_format == "html"

    malaysia = sources_by_doc_id["malaysia_pdpa_2010"]
    assert malaysia.download_mode == "auto"
    assert malaysia.preferred_format == "pdf"
    assert "UNDANG-UNDANG-MALAYSIA" in malaysia.url

    uk = sources_by_doc_id["uk_data_protection_act_2018"]
    assert uk.download_mode == "auto"
    assert uk.preferred_format == "xml"
    assert uk.url.endswith("/data.xml")

    japan = sources_by_doc_id["japan_appi_current_en"]
    assert japan.download_mode == "auto"
    assert japan.translation_status == "authoritative_translation"
    assert "japaneselawtranslation.go.jp" in japan.url

    korea = sources_by_doc_id["korea_pipa_current_en"]
    assert korea.download_mode == "auto"
    assert korea.preferred_format == "pdf"
    assert "fileDown.do" in korea.url

    south_africa = sources_by_doc_id["south_africa_popia_act_2013"]
    assert south_africa.download_mode == "auto"
    assert south_africa.preferred_format == "pdf"
    assert south_africa.url.endswith("3706726-11act4of2013popi.pdf")

    philippines = sources_by_doc_id["philippines_data_privacy_act_2012"]
    assert philippines.download_mode == "auto"
    assert philippines.preferred_format == "html"
    assert "lawphil.net" in philippines.url

    california = sources_by_doc_id["us_ca_ccpa_cpra_civ_1798_100_199"]
    assert california.download_mode == "auto"
    assert california.preferred_format == "html"
    assert "title=1.81.5." in california.url
