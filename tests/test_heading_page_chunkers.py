"""Tests for HeadingChunker and PageChunker (handbook 3.3, 3.4)."""

import random

import pytest

from rag_titan.chunking import HeadingChunker, PageChunker
from rag_titan.chunking.structure_aware import split_sections

MD = (
    "Intro text.\n\n"
    "# Leave policy\n\n"
    "## Annual leave\n\nStaff get twenty days.\n\n"
    "```python\n# not a heading\n```\n\n"
    "### Carry over\n\nUnused days move.\n\n"
    "## Sick leave\n\nTell your manager.\n\n"
    "# Travel\n\nBook early."
)


def test_split_sections_builds_nested_paths_and_skips_code():
    paths = [path for path, _ in split_sections(MD)]
    assert paths == [
        [],
        ["Leave policy", "Annual leave"],
        ["Leave policy", "Annual leave", "Carry over"],
        ["Leave policy", "Sick leave"],
        ["Travel"],
    ]
    assert "# not a heading" in split_sections(MD)[1][1]


def test_heading_chunker_prefixes_the_path():
    chunks = HeadingChunker(chunk_size=200).split_text(MD)
    assert chunks[0] == "Intro text."
    assert chunks[2] == "Leave policy > Annual leave > Carry over\n\nUnused days move."
    assert chunks[-1] == "Travel\n\nBook early."


def test_heading_chunker_without_path_keeps_heading_lines():
    chunks = HeadingChunker(chunk_size=200, include_path=False).split_text(MD)
    assert chunks[-1] == "# Travel\n\nBook early."
    # "# Leave policy" has no text of its own, so it joins the next section
    assert chunks[1].startswith("# Leave policy\n\n## Annual leave\n\nStaff get twenty days.")
    assert not any(chunk.strip() == "# Leave policy" for chunk in chunks)


def test_heading_path_counts_toward_chunk_size():
    text = "# A long section title\n\n" + "word " * 60
    chunks = HeadingChunker(chunk_size=60).split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert all(chunk.startswith("A long section title\n\n") for chunk in chunks)


def test_very_long_path_is_dropped_instead_of_crowding_out_text():
    text = "# " + "Title " * 10 + "\n\nShort body."
    assert HeadingChunker(chunk_size=40).split_text(text) == ["Short body."]


@pytest.mark.parametrize("seed", range(30))
def test_heading_chunks_never_mix_sections(seed):
    rng = random.Random(seed)
    parts = []
    for i in range(rng.randint(1, 8)):
        parts.append("#" * rng.randint(1, 3) + f" S{i}\n\n")
        parts.append(" ".join(f"s{i}w{j}." for j in range(rng.randint(0, 30))) + "\n\n")
    size = rng.randint(40, 200)
    for chunk in HeadingChunker(chunk_size=size).split_text("".join(parts)):
        assert len(chunk) <= size
        body = chunk.split("\n\n", 1)[-1]
        assert len({word.split("w")[0] for word in body.split()}) == 1   # one section only


def test_page_chunker_never_crosses_a_page():
    text = "Page one text.\fPage two is longer and has more words in it.\f\f   \fLast."
    chunks = PageChunker(chunk_size=30).split_text(text)
    assert chunks == ["Page one text.", "Page two is longer and has", "more words in it.", "Last."]


def test_page_chunker_without_page_breaks_is_one_page():
    assert PageChunker(chunk_size=100).split_text("One.\n\nTwo.") == ["One.\n\nTwo."]


@pytest.mark.parametrize("cls", [HeadingChunker, PageChunker])
def test_blank_text_and_bad_settings(cls):
    assert cls(chunk_size=10, overlap=0).split_text(" \n\f ") == []
    with pytest.raises(ValueError):
        cls(chunk_size=10, overlap=10)
