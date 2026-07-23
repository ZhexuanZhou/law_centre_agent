from pathlib import Path

from crawler.law_corpus.extract_text import (
    extract_text_from_file,
    extract_uk_legislation_metadata_from_file,
)


def test_extract_text_from_txt(tmp_path: Path):
    path = tmp_path / "law.txt"
    path.write_text("第一条 为了保护个人信息权益。", encoding="utf-8")

    assert extract_text_from_file(path) == "第一条 为了保护个人信息权益。"


def test_extract_text_from_html(tmp_path: Path):
    path = tmp_path / "law.html"
    path.write_text(
        "<html><head><title>x</title></head><body><h1>Article 1</h1><p>This Regulation protects people.</p></body></html>",
        encoding="utf-8",
    )

    text = extract_text_from_file(path)

    assert "Article 1" in text
    assert "This Regulation protects people." in text
    assert "<p>" not in text


def test_extract_text_from_australian_epub_html_skips_toc_and_endnotes(tmp_path: Path):
    path = tmp_path / "privacy_act.html"
    path.write_text(
        """<html><body>
<p class="TOC5">26C Eligible data breach</p>
<p id="navPoint_2" class="ActHead5">1 Short title</p>
<p class="subsection">This Act may be cited as the Privacy Act 1988.</p>
<p id="navPoint_3" class="ActHead5">6 Interpretation</p>
<p class="Definition">eligible data breach has the meaning given by section 26WE.</p>
<p id="navPoint_409" class="ActHead1">Schedule 1 — Australian Privacy Principles</p>
<p id="navPoint_410" class="ActHead5">1 Australian Privacy Principle 1—open and transparent management of personal information</p>
<p class="subsection">1.1 The object of this principle is transparency.</p>
<p id="navPoint_426" class="ActHead1">Schedule 2 — Statutory Tort for Serious Invasions of Privacy</p>
<p id="navPoint_427" class="ActHead5">1 Objects of this Schedule</p>
<p class="subsection">This schedule is not an Act section.</p>
<p class="ENotesHeading1">Endnotes</p>
<p class="ENoteTableText">26C amendment history</p>
</body></html>""",
        encoding="utf-8",
    )

    text = extract_text_from_file(path)

    assert text.startswith("1\nShort title\nThis Act may be cited as the Privacy Act 1988.")
    assert (
        "\n6\nInterpretation\neligible data breach has the meaning given by section 26WE." in text
    )
    assert "\nSchedule\n1\n—\nAustralian Privacy Principles\n" in text
    assert (
        "\n1\nAustralian Privacy Principle\n1—open and transparent management of personal information\n1.1\nThe object of this principle is transparency."
        in text
    )
    assert "TOC5" not in text
    assert "26C Eligible data breach" not in text
    assert "26C amendment history" not in text


def test_extract_text_from_xml(tmp_path: Path):
    path = tmp_path / "part.xml"
    path.write_text(
        """<?xml version="1.0"?><DIV5><HEAD>PART 312-CHILDREN'S ONLINE PRIVACY PROTECTION RULE</HEAD><SECTION><SECTNO>§ 312.1</SECTNO><SUBJECT>Scope of regulations.</SUBJECT></SECTION></DIV5>""",
        encoding="utf-8",
    )

    text = extract_text_from_file(path)

    assert "PART 312" in text
    assert "§ 312.1" in text
    assert "<HEAD>" not in text


def test_extract_text_from_legislation_gov_xml_keeps_sections_and_schedules_only(tmp_path: Path):
    path = tmp_path / "uk_dpa.xml"
    path.write_text(
        """<?xml version="1.0"?>
<Legislation xmlns="http://www.legislation.gov.uk/namespaces/legislation" xmlns:ukm="http://www.legislation.gov.uk/namespaces/metadata">
  <ukm:Metadata>
    <ukm:UnappliedEffect>
      <ukm:AffectedProvisions>
        <ukm:Section Ref="section-205-2-l">s. 205(2)(l)</ukm:Section>
      </ukm:AffectedProvisions>
    </ukm:UnappliedEffect>
  </ukm:Metadata>
  <Primary>
    <Body>
      <P1group>
        <Title>Overview</Title>
        <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/section/1" id="section-1">
          <Pnumber>1</Pnumber>
          <P1para>
            <P2><Pnumber>1</Pnumber><P2para><Text>This Act makes provision about personal data.</Text></P2para></P2>
          </P1para>
        </P1>
      </P1group>
      <P1group>
        <Title>Protection of personal data</Title>
        <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/section/2" id="section-2">
          <Pnumber>2</Pnumber>
          <P1para><Text>The UK GDPR and this Act protect individuals.</Text></P1para>
        </P1>
      </P1group>
    </Body>
    <Schedules>
      <Schedule DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/schedule/1">
        <Number>Schedule 1</Number>
        <TitleBlock><Title>Exemptions</Title></TitleBlock>
        <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/schedule/1/paragraph/1">
          <Pnumber>1</Pnumber><P1para><Text>Schedule paragraph should not be top-level section text.</Text></P1para>
        </P1>
      </Schedule>
      <Schedule DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/schedule/12A">
        <Number>Schedule 12A</Number>
        <TitleBlock><Title>The Information Commission</Title></TitleBlock>
        <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/schedule/12A/paragraph/1">
          <Pnumber>1</Pnumber><P1para><Text>The Commission is a body corporate.</Text></P1para>
        </P1>
      </Schedule>
    </Schedules>
    <BlockAmendment>
      <Schedule>
        <Number>Schedule 2</Number>
        <TitleBlock><Title>External instrument schedule</Title></TitleBlock>
        <P1 DocumentURI="http://www.legislation.gov.uk/uksi/2016/696/schedule/2/paragraph/1">
          <Pnumber>1</Pnumber><P1para><Text>Nested amendment schedule should not be extracted.</Text></P1para>
        </P1>
      </Schedule>
    </BlockAmendment>
  </Primary>
</Legislation>""",
        encoding="utf-8",
    )

    text = extract_text_from_file(path)

    assert text.startswith("1\nOverview\n(1) This Act makes provision about personal data.")
    assert "\n2\nProtection of personal data\nThe UK GDPR and this Act protect individuals." in text
    assert (
        "\nSCHEDULE 1\nExemptions\nParagraph 1\nSchedule paragraph should not be top-level section text."
        in text
    )
    assert (
        "\nSCHEDULE 12A\nThe Information Commission\nParagraph 1\nThe Commission is a body corporate."
        in text
    )
    assert "s. 205(2)(l)" not in text
    assert "Nested amendment schedule" not in text


def test_extract_uk_legislation_metadata_preserves_section_revision_status(tmp_path: Path):
    path = tmp_path / "uk_dpa.xml"
    path.write_text(
        """<?xml version="1.0"?>
<Legislation xmlns="http://www.legislation.gov.uk/namespaces/legislation">
  <Primary><Body>
    <P1group RestrictStartDate="2025-08-20">
      <Title>Meaning of “court”</Title>
      <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/section/20">
        <Pnumber>20</Pnumber><P1para><Text>. . . . . . . .</Text><CommentaryRef Ref="c20" /></P1para>
      </P1>
    </P1group>
    <P1group RestrictStartDate="2025-08-20">
      <Title><Addition CommentaryRef="c114a">The Information Commission</Addition></Title>
      <P1 DocumentURI="http://www.legislation.gov.uk/ukpga/2018/12/section/114A">
        <Pnumber><Addition CommentaryRef="c114a">114A</Addition></Pnumber>
        <P1para><Text><Addition CommentaryRef="c114a">The Commission is established.</Addition></Text></P1para>
      </P1>
    </P1group>
  </Body></Primary>
  <Commentaries>
    <Commentary id="c20">S. 20 omitted (20.8.2025) by virtue of another Act.</Commentary>
    <Commentary id="c114a">S. 114A and cross-heading inserted (20.8.2025) by another Act.</Commentary>
  </Commentaries>
</Legislation>""",
        encoding="utf-8",
    )

    metadata = extract_uk_legislation_metadata_from_file(path)
    sections = metadata["sections"]

    assert sections["20"]["status"] == "omitted"
    assert sections["20"]["title"] == "Meaning of “court”"
    assert sections["20"]["effective_to"] == "2025-08-20"
    assert sections["20"]["commentaries"][0]["ref"] == "c20"
    assert sections["114A"]["status"] == "active"
    assert sections["114A"]["effective_from"] == "2025-08-20"
