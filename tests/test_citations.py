"""Tests for page and section in the answer context, end to end through the pipeline."""

from conftest import FakeEmbedder, FakeLLM

from rag_titan.chunking import HeadingChunker
from rag_titan.generation import GroundedGenerator, build_context
from rag_titan.generation.context_builder import describe_source
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import Retriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import MemoryStore


def test_header_without_location_is_unchanged():
    assert describe_source({"id": "chunk_0", "source": "a.pdf"}) == "[chunk_0] Source: a.pdf"


def test_header_with_page_range_and_section():
    chunk = {"id": "chunk_4", "source": "h.pdf", "page_numbers": [3],
             "section_path": ["Leave policy", "Sick leave"]}
    assert describe_source(chunk) == (
        "[chunk_4] Source: h.pdf, page 3, section: Leave policy > Sick leave")
    assert describe_source({"id": "c", "source": "h.pdf", "page_numbers": [3, 4]}) == (
        "[c] Source: h.pdf, pages 3-4")


def test_empty_section_path_is_left_out():
    chunk = {"id": "chunk_1", "source": "a.md", "section_path": []}
    assert describe_source(chunk) == "[chunk_1] Source: a.md"


def test_page_and_section_survive_index_search_rerank(tmp_path):
    path = tmp_path / "handbook.txt"   # a .txt with page breaks behaves like a PDF's text
    path.write_text("# Pay\n\nSalary is paid every month.\f"
                    "# Leave\n\n## Holiday\n\nYou get 20 leave days every year.", encoding="utf-8")
    embedder, store = FakeEmbedder(), MemoryStore()

    IndexingPipeline(HeadingChunker(200), embedder, store).run([str(path)])
    matches = Retriever(embedder, store).retrieve("how many leave days", top_k=2)
    best = KeywordReranker().rerank("how many leave days", matches, top_n=1)

    assert best[0]["page_numbers"] == [2]
    assert best[0]["section_path"] == ["Leave", "Holiday"]
    assert "Source: handbook.txt, page 2, section: Leave > Holiday" in build_context(best)


def test_generator_sends_page_citation_rule_and_page_to_the_llm():
    llm = FakeLLM()
    chunk = {"id": "chunk_0", "source": "h.pdf", "text": "20 leave days", "page_numbers": [3]}
    GroundedGenerator(llm).generate("how many leave days?", [chunk])

    system, user = llm.calls[-1][0]["content"], llm.calls[-1][1]["content"]
    assert "you may add it, such as [chunk_0, page 3]" in system
    assert "[chunk_0] Source: h.pdf, page 3" in user
