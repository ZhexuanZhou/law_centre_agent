from crawler.law_corpus.models import SourceDocument
from crawler.law_corpus.parsers.china import ChinaLegalParser, chinese_article_to_int


def test_china_parser_extracts_articles_with_title_on_next_line():
    doc = SourceDocument(
        doc_id="china_pipl_2021",
        jurisdiction="CN",
        law_family="china_pipl",
        source_type="primary_law",
        title="中华人民共和国个人信息保护法",
        version_date="2021-08-20",
        effective_date="2021-11-01",
        source_url="https://example.test",
        language="zh",
        raw_text=(
            "第一章 总则\n"
            "第一条\n"
            "为了保护个人信息权益，规范个人信息处理活动，制定本法。\n"
            "第二条\n"
            "自然人的个人信息受法律保护。\n"
        ),
    )

    units = ChinaLegalParser().parse(doc)

    assert [unit.local_citation for unit in units] == ["第一条", "第二条"]
    assert units[0].canonical_citation == "PIPL Article 1"
    assert "规范个人信息处理活动" in units[0].text
    assert units[0].span_ids[0].startswith("china_pipl_2021:span:")


def test_chinese_article_to_int_handles_compound_numbers():
    assert chinese_article_to_int("第一条") == 1
    assert chinese_article_to_int("第十条") == 10
    assert chinese_article_to_int("第十一条") == 11
    assert chinese_article_to_int("第二十条") == 20
    assert chinese_article_to_int("第二十一条") == 21
    assert chinese_article_to_int("第七十四条") == 74
