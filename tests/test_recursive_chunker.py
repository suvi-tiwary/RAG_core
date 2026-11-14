"""Tests for RecursiveChunker (handbook section 2.5)."""

import pytest

from rag_titan.chunking import RecursiveChunker

STORY = (
    "Leave policy. Staff get twenty days of paid leave every year.\n\n"
    "Sick days are separate. Tell your manager before nine on the first day.\n"
    "A doctor's note is needed after three days."
)


def no_spaces(text):
    """Text without any whitespace, to compare content only."""
    return "".join(text.split())


class WordTokenizer:
    """Counts one token per word, so token sizes are easy to check."""

    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_recursive_keeps_paragraphs_whole_when_they_fit():
    chunker = RecursiveChunker(chunk_size=12, overlap=0)
    assert chunker.split_text("para one.\n\npara two.") == ["para one.", "para two."]


def test_recursive_falls_back_to_sentences_then_hard_cut():
    chunker = RecursiveChunker(chunk_size=10, overlap=0)
    chunks = chunker.split_text("aaa. bbb. " + "x" * 25)
    assert chunks == ["aaa. bbb.", "x" * 10, "x" * 10, "x" * 5]


@pytest.mark.parametrize("size", [5, 20, 40, 80, 500])
def test_recursive_sizes_and_nothing_lost(size):
    chunks = RecursiveChunker(chunk_size=size, overlap=0).split_text(STORY)

    assert all(chunk and len(chunk) <= size for chunk in chunks)
    assert no_spaces("".join(chunks)) == no_spaces(STORY)


def test_recursive_repeats_whole_pieces_as_overlap():
    chunker = RecursiveChunker(chunk_size=10, overlap=5)
    assert chunker.split_text("aaa. bbb. ccc. ddd.") == ["aaa. bbb.", "bbb. ccc.", "ccc. ddd."]


def test_recursive_counts_tokens_with_own_tokenizer():
    chunker = RecursiveChunker(chunk_size=3, overlap=0, tokenizer=WordTokenizer())
    assert chunker.split_text("a b c d e f") == ["a b c", "d e f"]


def test_recursive_blank_text_gives_no_chunks():
    assert RecursiveChunker(chunk_size=10, overlap=0).split_text(" \n\n ") == []


@pytest.mark.parametrize("settings", [
    {"chunk_size": 10, "overlap": 10},
    {"chunk_size": 10, "overlap": 0, "separators": ()},
    {"chunk_size": 10, "overlap": 0, "tokenizer": "not a tokenizer"},
])
def test_recursive_rejects_bad_settings(settings):
    with pytest.raises(ValueError):
        RecursiveChunker(**settings)
