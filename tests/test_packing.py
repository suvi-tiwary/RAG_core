"""Tests for pack(): whole pieces up to chunk_size, whole-piece overlap up to `overlap`."""

from rag_titan.chunking.packing import join_groups, pack


def test_pack_fills_chunks_without_overlap():
    assert pack([3, 3, 3, 3], chunk_size=6) == [[0, 1], [2, 3]]


def test_pack_repeats_whole_pieces_as_overlap():
    assert pack([3, 3, 3, 3], chunk_size=6, overlap=3) == [[0, 1], [1, 2], [2, 3]]


def test_pack_never_repeats_a_piece_bigger_than_overlap():
    assert pack([5, 5, 5], chunk_size=10, overlap=4) == [[0, 1], [2]]


def test_pack_gives_an_oversized_piece_its_own_chunk():
    assert pack([2, 9, 2], chunk_size=5) == [[0], [1], [2]]


def test_pack_drops_repeated_pieces_that_no_longer_fit():
    # piece 1 (size 3) fits in the overlap, but 3 + 4 > 6, so it is dropped again
    assert pack([3, 3, 4], chunk_size=6, overlap=3) == [[0, 1], [2]]


def test_pack_empty_and_join():
    assert pack([], chunk_size=5) == []
    assert join_groups(["a ", "b ", "c"], [[0, 1], [1, 2]]) == ["a b ", "b c"]
