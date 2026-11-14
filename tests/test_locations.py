"""Tests for page_numbers and section_path on chunks."""

import pytest

from rag_titan.chunking import CharacterChunker, HeadingChunker, PageChunker, WordChunker
from rag_titan.chunking.headings import heading_marks
from rag_titan.chunking.locations import Locations

PDF = "Page one text.\fPage two text.\f\fPage four text."
MD = ("Intro.\n\n# Leave policy\n\n## Annual\n\nTwenty days.\n\n"
      "```\n# not a heading\n```\n\n## Sick\n\nCall.")


def test_page_numbers_inside_one_page_and_across_pages():
    locations = Locations(PDF)
    assert locations.describe(0, 14) == {"page_numbers": [1]}
    assert locations.describe(10, 20) == {"page_numbers": [1, 2]}   # crosses one page break
    assert locations.describe(31, 46) == {"page_numbers": [4]}       # page 3 is empty
    assert locations.describe(14, 15) == {"page_numbers": [1]}       # the break itself ends page 1


def test_heading_marks_build_nested_paths_and_skip_code():
    paths = [path for _, path in heading_marks(MD)]
    assert paths == [["Leave policy"], ["Leave policy", "Annual"], ["Leave policy", "Sick"]]


def test_section_path_is_the_heading_open_at_the_chunk_start():
    locations = Locations(MD)
    assert locations.describe(0, 5) == {"section_path": []}   # before the first heading
    assert locations.describe(MD.index("Twenty"), len(MD)) == {
        "section_path": ["Leave policy", "Annual"]}
    assert locations.describe(MD.index("# not"), len(MD))["section_path"] == [
        "Leave policy", "Annual"]                               # "#" in code is not a heading
    assert locations.describe(MD.index("Call"), len(MD))["section_path"] == [
        "Leave policy", "Sick"]


def test_plain_text_gets_no_extra_fields():
    chunk = WordChunker(5, 0).chunk([{"source": "a.txt", "text": "just some words"}])[0]
    assert "page_numbers" not in chunk and "section_path" not in chunk


def test_unknown_offsets_give_no_fields():
    assert Locations(PDF).describe(None, None) == {}


@pytest.mark.parametrize("chunker", [
    CharacterChunker(20, 5), WordChunker(4, 1), HeadingChunker(60), PageChunker(60)])
def test_every_chunker_gets_pages_and_sections(chunker):
    text = "# Leave\n\nStaff get twenty days.\f## Sick\n\nTell your manager today.\fEnd."
    for chunk in chunker.chunk([{"source": "h.pdf", "text": text}]):
        start = chunk["start_offset"]
        page = 1 + text[:start].count("\f")
        assert chunk["page_numbers"][0] == page
        assert chunk["section_path"][0] == "Leave"
        if page == 2 and start > text.index("## Sick"):
            assert chunk["section_path"] == ["Leave", "Sick"]


def test_pdf_with_headings_gets_both_fields():
    text = "# Leave\n\nText.\f## Sick\n\nMore."
    chunk = PageChunker(100).chunk([{"source": "h.pdf", "text": text}])[1]
    assert chunk["page_numbers"] == [2]
    assert chunk["section_path"] == ["Leave", "Sick"]
