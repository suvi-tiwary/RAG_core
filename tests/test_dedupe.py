"""Tests for removing near-copies from search results (handbook 2.4)."""

import pytest
from conftest import FakeEmbedder

from rag_titan import get_settings
from rag_titan.chunking import WordChunker
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import Retriever, overlap_fraction, remove_duplicates
from rag_titan.vector_stores import MemoryStore


def hit(name, start, end, source="a.pdf", text=None):
    return {"id": name, "source": source, "text": text or name,
            "start_offset": start, "end_offset": end}


def test_overlap_fraction():
    assert overlap_fraction(hit("a", 0, 100), hit("b", 50, 150)) == 0.5    # half shared
    assert overlap_fraction(hit("a", 0, 100), hit("b", 20, 40)) == 1.0     # b inside a
    assert overlap_fraction(hit("a", 0, 100), hit("b", 100, 200)) == 0.0   # touching only
    assert overlap_fraction(hit("a", 0, 100), hit("b", 0, 100, source="b.pdf")) == 0.0
    assert overlap_fraction({"source": "a.pdf"}, hit("b", 0, 100)) == 0.0  # no offsets


def test_remove_duplicates_keeps_the_better_hit_and_the_order():
    matches = [hit("best", 0, 100), hit("near_copy", 30, 130), hit("other", 300, 400),
               hit("small_overlap", 380, 480), hit("other_file", 0, 100, source="b.pdf")]
    kept = remove_duplicates(matches, min_overlap=0.5)
    assert [m["id"] for m in kept] == ["best", "other", "small_overlap", "other_file"]


def test_chunks_without_offsets_are_copies_only_when_the_text_is_the_same():
    a = {"id": "a", "source": "x.txt", "text": "same words"}
    b = {"id": "b", "source": "x.txt", "text": "same words"}
    c = {"id": "c", "source": "x.txt", "text": "other words"}
    d = {"id": "d", "source": "y.txt", "text": "same words"}
    assert [m["id"] for m in remove_duplicates([a, b, c, d])] == ["a", "c", "d"]


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_bad_min_overlap_is_rejected(bad):
    with pytest.raises(ValueError):
        remove_duplicates([], min_overlap=bad)
    with pytest.raises(ValueError):
        Retriever(FakeEmbedder(), MemoryStore(), dedupe_overlap=bad)


def index(text, chunker):
    embedder, store = FakeEmbedder(), MemoryStore()
    chunks = chunker.chunk([{"source": "hr.txt", "text": text}])
    store.add(chunks, embedder.embed_many([c["text"] for c in chunks]))
    return embedder, store


TEXT = ("leave days leave days leave days office work remote work salary salary "
        "holiday leave days remote office work salary holiday leave days leave")


def test_retriever_returns_different_chunks_and_still_fills_top_k():
    embedder, store = index(TEXT, WordChunker(chunk_size=6, overlap=4))   # heavy overlap
    results = Retriever(embedder, store).retrieve("leave days", top_k=3)

    assert len(results) == 3
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            assert overlap_fraction(a, b) < 0.5


def test_dedupe_off_is_the_old_behaviour():
    embedder, store = index(TEXT, WordChunker(chunk_size=6, overlap=4))
    old = store.search(embedder.embed("leave days"), top_k=3)
    new = Retriever(embedder, store, dedupe_overlap=None).retrieve("leave days", top_k=3)
    # same chunks, order and scores (0.6.0 only adds "retriever" and "score_trace")
    assert [(m["id"], m["score"]) for m in new] == [(m["id"], m["score"]) for m in old]


def test_setting_reads_share_or_off(monkeypatch):
    assert get_settings()["dedupe_overlap"] == 0.5
    monkeypatch.setenv("DEDUPE_OVERLAP", "off")
    assert get_settings()["dedupe_overlap"] is None
    monkeypatch.setenv("DEDUPE_OVERLAP", "0.3")
    assert get_settings()["dedupe_overlap"] == 0.3
    monkeypatch.setenv("DEDUPE_OVERLAP", "lots")
    with pytest.raises(ValueError):
        get_settings()


def test_pipeline_context_no_longer_holds_the_same_passage_twice(tmp_path):
    path = tmp_path / "hr.txt"
    path.write_text(TEXT, encoding="utf-8")
    embedder, store = FakeEmbedder(), MemoryStore()
    IndexingPipeline(WordChunker(chunk_size=6, overlap=4), embedder, store).run([str(path)])

    with_dupes = Retriever(embedder, store, dedupe_overlap=None).retrieve("leave days", top_k=4)
    without = Retriever(embedder, store).retrieve("leave days", top_k=4)

    pairs = [(a, b) for i, a in enumerate(with_dupes) for b in with_dupes[i + 1:]]
    assert any(overlap_fraction(a, b) >= 0.5 for a, b in pairs)   # the problem is real
    assert all(overlap_fraction(a, b) < 0.5
               for i, a in enumerate(without) for b in without[i + 1:])


def test_factory_passes_the_setting_to_the_retriever():
    from rag_titan.factory import build_rag_parts

    assert build_rag_parts(get_settings())["retriever"].dedupe_overlap == 0.5
    parts = build_rag_parts(get_settings({"dedupe_overlap": None}))
    assert parts["retriever"].dedupe_overlap is None
