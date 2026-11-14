"""Tests for chunk offsets: start_offset / end_offset point at the chunk in the document."""

import random

import pytest

from rag_titan.chunking import (
    BaseChunker,
    CharacterChunker,
    HeadingChunker,
    PageChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
)
from rag_titan.chunking.offsets import locate


class WordTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def random_document(seed):
    rng = random.Random(seed)
    parts = ["Leave", "policy", "Yes.", "Yes.", "Dr.", "3.5", "über", "日本語。", "x" * 40,
             "\n\n# Title\n\n", "\n\n## Sub\n\n", "\f", "\n```\n# code\n```\n", "\n\n", "\t"]
    seps = [" ", " ", ". ", "\n", ""]
    return "".join(rng.choice(parts) + rng.choice(seps) for _ in range(rng.randint(1, 80)))


def all_chunkers(size, overlap):
    return [CharacterChunker(size, overlap), WordChunker(size, overlap),
            RecursiveChunker(size, overlap), SentenceChunker(size, overlap),
            ParagraphChunker(size, overlap), HeadingChunker(size, overlap),
            HeadingChunker(size, overlap, include_path=False), PageChunker(size, overlap),
            TokenChunker(size, overlap, tokenizer=WordTokenizer())]


def test_locate_finds_exact_and_respaced_text_in_order():
    text = "one  two\nthree one two"
    assert locate(text, ["one  two", "one two", "missing"]) == [(0, 8), (15, 22), (None, None)]


@pytest.mark.parametrize("seed", range(60))
def test_offsets_point_at_the_chunk(seed):
    text = random_document(seed)
    size = random.Random(seed).randint(3, 120)
    overlap = random.Random(seed + 1).randint(0, size - 1)

    for chunker in all_chunkers(size, overlap):
        chunks = chunker.chunk([{"source": "doc.md", "text": text}])
        starts = [chunk["start_offset"] for chunk in chunks]
        # in order; two chunks can start at the same place when one is the start of the other
        assert starts == sorted(starts), type(chunker).__name__
        for chunk in chunks:
            start, end = chunk["start_offset"], chunk["end_offset"]
            assert 0 <= start < end <= len(text)
            source = text[start:end]
            if isinstance(chunker, (WordChunker, TokenChunker)):
                assert source.split() == chunk["text"].split()   # same words, original spacing
            elif isinstance(chunker, CharacterChunker):
                assert source == chunk["text"]   # exact text, spaces at the edges included
            else:
                assert chunk["text"].endswith(source)   # heading chunks have the path in front
                assert source.strip() == source   # packing chunkers trim the edges


def test_repeated_text_gets_the_true_positions():
    def starts(chunker, text):
        return [chunk["start_offset"] for chunk in chunker.chunk([{"source": "a", "text": text}])]

    assert starts(CharacterChunker(6, 2), "ab" * 30) == list(range(0, 57, 4))

    assert starts(SentenceChunker(20, 5), "Yes. " * 12) == [0, 15, 30, 45]
    assert starts(PageChunker(100), "Same footer line.\f" * 4) == [0, 18, 36, 54]


def test_heading_offsets_point_at_the_section_text():
    text = "# Leave\n\n## Sick\n\nTell your manager."
    chunk = HeadingChunker(100).chunk([{"source": "a.md", "text": text}])[0]
    assert chunk["text"] == "Leave > Sick\n\nTell your manager."
    assert text[chunk["start_offset"]:chunk["end_offset"]] == "Tell your manager."


class UpperChunker(BaseChunker):
    """A chunker someone wrote, that changes the text (so it cannot be found)."""

    def split_text(self, text):
        return [text.upper(), text]


def test_custom_chunkers_get_offsets_or_none():
    chunks = UpperChunker().chunk([{"source": "a", "text": "hello world"}])
    assert (chunks[0]["start_offset"], chunks[0]["end_offset"]) == (None, None)
    assert (chunks[1]["start_offset"], chunks[1]["end_offset"]) == (0, 11)


def test_chunk_ids_and_sources_are_unchanged():
    chunks = WordChunker(2, 0).chunk([{"source": "a.pdf", "text": "one two"},
                                      {"source": "b.pdf", "text": "three"}])
    assert [(c["id"], c["source"]) for c in chunks] == [("chunk_0", "a.pdf"), ("chunk_1", "b.pdf")]
    assert (chunks[1]["start_offset"], chunks[1]["end_offset"]) == (0, 5)
