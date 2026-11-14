"""Tests for fixed-length chunking (handbook sections 2.1, 2.2, 2.4 and 12)."""

import pytest

from rag_titan.chunking import CharacterChunker, WordChunker, count_windows, fixed_windows

TEXT = "Fixed-length chunking cuts text into windows.\nEach window overlaps the last."


def rebuild(windows, overlap):
    """Join windows back together, dropping the overlapping part of each one."""
    joined = windows[0]
    for window in windows[1:]:
        joined = joined + window[overlap:]
    return joined


# --- the shared helper ---

def test_fixed_windows_handbook_examples():
    assert fixed_windows([], 3, 0) == []
    assert fixed_windows([1, 2, 3, 4, 5], 3, 1) == [[1, 2, 3], [3, 4, 5]]


@pytest.mark.parametrize("size, overlap", [(3, 3), (3, 4), (0, 0), (3, -1), (2.5, 0)])
def test_fixed_windows_rejects_bad_settings(size, overlap):
    with pytest.raises(ValueError):
        fixed_windows([1, 2], size, overlap)


@pytest.mark.parametrize("size, overlap", [(1, 0), (4, 1), (7, 3), (10, 0), (200, 50)])
def test_fixed_windows_exact_sizes_and_lossless(size, overlap):
    windows = fixed_windows(TEXT, size, overlap)

    assert all(windows)
    assert all(len(window) == size for window in windows[:-1])
    assert rebuild(windows, overlap) == TEXT


@pytest.mark.parametrize("size, overlap", [(1, 0), (3, 1), (5, 4), (8, 0), (8, 3)])
def test_count_windows_matches_fixed_windows(size, overlap):
    for n in range(0, 40):
        units = list(range(n))
        assert count_windows(n, size, overlap) == len(fixed_windows(units, size, overlap))


def test_count_windows_shows_overlap_cost():
    # 1000 units in windows of 100: 10 without overlap, 13 with 20 overlap
    assert count_windows(1000, 100, 0) == 10
    assert count_windows(1000, 100, 20) == 13


def test_count_windows_rejects_bad_n():
    with pytest.raises(ValueError):
        count_windows(-1, 3, 0)


# --- character chunker ---

def test_character_chunker_keeps_text_exactly():
    chunker = CharacterChunker(chunk_size=4, overlap=1)
    assert chunker.split_text("abcdefghij") == ["abcd", "defg", "ghij"]


def test_character_chunker_blank_text_gives_no_chunks():
    assert CharacterChunker(chunk_size=5, overlap=0).split_text("  \n ") == []


def test_character_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        CharacterChunker(chunk_size=5, overlap=5)


def test_character_chunker_works_with_chunk():
    chunker = CharacterChunker(chunk_size=3, overlap=0)
    chunks = chunker.chunk([{"source": "a.txt", "text": "abcdef"}])
    assert [chunk["text"] for chunk in chunks] == ["abc", "def"]
    assert chunks[1]["source"] == "a.txt"


# --- word chunker on the shared helper ---

def test_word_chunker_lossless_over_words():
    chunker = WordChunker(chunk_size=3, overlap=1)
    windows = [piece.split() for piece in chunker.split_text(TEXT)]
    assert rebuild(windows, 1) == TEXT.split()


def test_old_word_chunker_import_path_still_works():
    # code written for 0.1.0 imports from rag_titan.chunking.word_chunker
    from rag_titan.chunking.word_chunker import WordChunker as OldWordChunker

    assert OldWordChunker is WordChunker
