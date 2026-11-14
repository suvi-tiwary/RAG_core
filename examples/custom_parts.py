"""
Build a RAG by hand, choosing every part yourself.

This shows how the building blocks fit together, so you can swap
any of them for your own class.

Run:
    python examples/custom_parts.py path/to/file.pdf
"""

import sys

from rag_titan import SimpleRAG
from rag_titan.chunking import WordChunker
from rag_titan.embeddings import OllamaEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.llm import OllamaLLM
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import Retriever
from rag_titan.retrieval.rerankers import NoReranker
from rag_titan.vector_stores import MemoryStore


def main():
    """Index one file with hand-picked parts and ask a question."""
    llm = OllamaLLM("gemma3:4b")
    embedder = OllamaEmbedder("nomic-embed-text")
    store = MemoryStore()

    # 1. indexing: load -> chunk -> embed -> store
    pipeline = IndexingPipeline(WordChunker(chunk_size=200, overlap=40), embedder, store)
    pipeline.run(sys.argv[1:])

    # 2. answering: retrieve -> rerank -> generate
    rag = SimpleRAG(
        retriever=Retriever(embedder, store),
        reranker=NoReranker(),
        generator=GroundedGenerator(llm),
        top_k=6,
        top_n=3,
    )
    print(rag.answer("Summarise the main points."))


if __name__ == "__main__":
    main()
