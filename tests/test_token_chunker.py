"""Tests for TokenChunker (handbook section 2.3). No tiktoken or network needed."""

import sys

import pytest

from rag_titan.chunking import TokenChunker


class FakeTokenizer:
    """One token per word, so the expected chunks are easy to read."""

    def encode(self, text):
        self.vocab = text.split()
        return list(range(len(self.vocab)))

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)


def test_token_chunker_counts_tokens_with_own_tokenizer():
    chunker = TokenChunker(chunk_size=3, overlap=1, tokenizer=FakeTokenizer())
    assert chunker.split_text("a b c d e f g") == ["a b c", "c d e", "e f g"]


def test_token_chunker_blank_text_gives_no_chunks():
    chunker = TokenChunker(chunk_size=3, overlap=0, tokenizer=FakeTokenizer())
    assert chunker.split_text("   ") == []


def test_token_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=4, overlap=4, tokenizer=FakeTokenizer())


def test_token_chunker_rejects_tokenizer_without_methods():
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=4, overlap=0, tokenizer="not a tokenizer")


def test_token_chunker_explains_missing_tiktoken(monkeypatch):
    # None in sys.modules makes `import tiktoken` fail, even if it is installed
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    with pytest.raises(ImportError, match="rag-titan\\[tokens\\]"):
        TokenChunker(chunk_size=4, overlap=0)
