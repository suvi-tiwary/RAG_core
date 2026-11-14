"""Tests for SentenceChunker and ParagraphChunker (handbook 3.1, 3.2)."""

import random

import pytest

from rag_titan.chunking import ParagraphChunker, SentenceChunker
from rag_titan.chunking.structure_aware import split_paragraphs, split_sentences

DOC = (
    "# Leave policy\n\n"
    "Staff get twenty days. Unused days carry over. Ask HR for the form.\n\n"
    "Sick leave is separate. Tell your manager before nine."
)


def no_spaces(text):
    return "".join(text.split())


def random_text(seed):
    rng = random.Random(seed)
    words = ["leave", "Policy", "staff", "Dr.", "3.5", "über", "a", "x" * 40]
    ends = [" ", " ", ". ", "! ", ".\n", "\n\n"]
    return "".join(rng.choice(words) + rng.choice(ends) for _ in range(rng.randint(0, 60)))


class WordTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_sentence_chunker_packs_whole_sentences():
    chunks = SentenceChunker(chunk_size=50, overlap=0).split_text(DOC)
    assert chunks == [
        "# Leave policy\n\nStaff get twenty days.",
        "Unused days carry over. Ask HR for the form.",
        "Sick leave is separate.",
        "Tell your manager before nine.",
    ]


def test_sentence_chunker_overlap_repeats_whole_sentences():
    chunks = SentenceChunker(chunk_size=50, overlap=25).split_text(DOC)
    assert chunks[1] == "Staff get twenty days. Unused days carry over."


def test_paragraph_chunker_keeps_paragraphs_and_splits_big_ones():
    chunks = ParagraphChunker(chunk_size=60, overlap=0).split_text(DOC)
    # paragraph 2 (69 chars) is too big, so it is cut into sentences:
    # 16 + 23 = 39 fits, + 24 = 63 > 60, so a new chunk starts
    assert chunks == [
        "# Leave policy\n\nStaff get twenty days.",
        "Unused days carry over. Ask HR for the form.",
        "Sick leave is separate. Tell your manager before nine.",
    ]


@pytest.mark.parametrize("cls, units", [
    (SentenceChunker, split_sentences),
    (ParagraphChunker, split_paragraphs),
])
@pytest.mark.parametrize("seed", range(40))
def test_sizes_nothing_lost_and_units_kept_whole(cls, units, seed):
    text = random_text(seed)
    size = random.Random(seed).randint(5, 120)
    chunks = cls(chunk_size=size, overlap=0).split_text(text)

    assert all(chunk and len(chunk) <= size for chunk in chunks)
    assert no_spaces("".join(chunks)) == no_spaces(text)
    for unit in units(text):
        if unit.strip() and len(unit) <= size:   # a unit that fits is never cut
            assert any(unit.strip() in chunk for chunk in chunks)


def test_long_sentence_falls_back_to_words():
    chunks = SentenceChunker(chunk_size=12, overlap=0).split_text("one two three four five six.")
    assert chunks == ["one two", "three four", "five six."]


def test_tokenizer_counts_tokens():
    chunker = SentenceChunker(chunk_size=6, overlap=0, tokenizer=WordTokenizer())
    chunks = chunker.split_text("Ann was here. Bob came too. Go now.")
    assert chunks == ["Ann was here. Bob came too.", "Go now."]   # 3 + 3 tokens, then 2


@pytest.mark.parametrize("cls", [SentenceChunker, ParagraphChunker])
def test_blank_text_and_bad_settings(cls):
    assert cls(chunk_size=10, overlap=0).split_text(" \n\n ") == []
    with pytest.raises(ValueError):
        cls(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        cls(chunk_size=10, overlap=0, tokenizer="nope")
