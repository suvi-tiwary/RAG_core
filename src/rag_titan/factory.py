"""
The only place that knows which class to use for each setting.

Adding a new provider or strategy:
    1. create its class in the right folder
    2. add one `if name == ...` line in the matching build_* function
    3. change the setting in .env (or pass it in `get_settings()`)
"""

from rag_titan.chunking.fixed_size import (
    CharacterChunker,
    RecursiveChunker,
    TokenChunker,
    WordChunker,
)
from rag_titan.chunking.structure_aware import (
    HeadingChunker,
    PageChunker,
    ParagraphChunker,
    SentenceChunker,
)
from rag_titan.embeddings.hash_embedder import HashEmbedder
from rag_titan.embeddings.ollama_embedder import OllamaEmbedder
from rag_titan.generation.grounded_generator import GroundedGenerator
from rag_titan.llm.ollama_llm import OllamaLLM
from rag_titan.pipelines.indexing_pipeline import IndexingPipeline
from rag_titan.rag_types.adaptive.adaptive_rag import AdaptiveRAG
from rag_titan.rag_types.simple_rag import SimpleRAG
from rag_titan.retrieval.dense_retriever import DenseRetriever
from rag_titan.retrieval.rerankers.keyword_reranker import KeywordReranker
from rag_titan.retrieval.rerankers.no_reranker import NoReranker
from rag_titan.vector_stores.memory_store import MemoryStore

# every RAG type the package can build, used by build_rag() and the CLI
RAG_TYPES = ["simple", "adaptive"]

# every chunker build_chunker() knows, used by the CLI
CHUNKERS = ["word", "character", "token", "recursive", "sentence", "paragraph", "heading", "page"]


def build_llm(settings, model=None, think=False):
    """Create the chat model named in settings["llm_provider"]."""
    name = settings["llm_provider"]
    model = model or settings["llm_model"]

    if name == "ollama":
        return OllamaLLM(model, host=settings["ollama_host"], think=think)

    raise ValueError(f"Unknown LLM provider: {name}")


def build_agent_llm(settings):
    """Create the chat model used by agents (it must support tools)."""
    return build_llm(settings, model=settings["agent_model"], think=settings["agent_think"])


def build_embedder(settings):
    """Create the embedding model named in settings["embed_provider"]."""
    name = settings["embed_provider"]

    if name == "ollama":
        return OllamaEmbedder(settings["embed_model"], host=settings["ollama_host"])

    if name == "hash":
        return HashEmbedder()

    raise ValueError(f"Unknown embedding provider: {name}")


def build_chunker(settings):
    """
    Create the chunker named in settings["chunker"].
    chunk_size and chunk_overlap count words, characters or tokens,
    depending on the chunker.
    """
    name = settings["chunker"]
    size, overlap = settings["chunk_size"], settings["chunk_overlap"]

    if name == "word":
        return WordChunker(size, overlap)

    if name == "character":
        return CharacterChunker(size, overlap)

    if name == "token":
        return TokenChunker(size, overlap, encoding=settings["chunk_encoding"])

    if name == "recursive":
        return RecursiveChunker(size, overlap)

    # structure-aware chunkers: whole sentences, paragraphs, sections or pages
    structure_aware = {
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "heading": HeadingChunker,
        "page": PageChunker,
    }
    if name in structure_aware:
        return structure_aware[name](size, overlap)

    raise ValueError(f"Unknown chunker: {name}")


def build_store(settings):
    """Create the vector store named in settings["vector_store"]."""
    name = settings["vector_store"]

    if name == "memory":
        return MemoryStore()

    raise ValueError(f"Unknown vector store: {name}")


def build_reranker(settings, llm=None):
    """
    Create the reranker named in settings["reranker"].

    `llm` is not used yet. It is here for future LLM-based rerankers.
    """
    name = settings["reranker"]

    if name == "keyword":
        return KeywordReranker()
    if name == "none":
        return NoReranker()

    raise ValueError(f"Unknown reranker: {name}")


def build_generator(settings, llm):
    """Create the answer generator named in settings["generator"]."""
    name = settings["generator"]

    if name == "grounded":
        return GroundedGenerator(llm)

    raise ValueError(f"Unknown generator: {name}")


def build_rag_parts(settings):
    """
    Build every RAG building block, sharing one embedder and one store.

    Returns:
        A dict with "llm", "pipeline", "retriever", "reranker"
        and "generator".
    """
    llm = build_llm(settings)
    embedder = build_embedder(settings)
    store = build_store(settings)

    return {
        "llm": llm,
        "pipeline": IndexingPipeline(build_chunker(settings), embedder, store),
        "retriever": DenseRetriever(embedder, store, dedupe_overlap=settings["dedupe_overlap"]),
        "reranker": build_reranker(settings, llm),
        "generator": build_generator(settings, llm),
    }


def build_rag(rag_type, parts):
    """
    Create one RAG type from the parts made by build_rag_parts().

    Args:
        rag_type: one of RAG_TYPES, for example "simple" or "adaptive".
        parts: the dict returned by build_rag_parts().
    """
    if rag_type == "simple":
        return SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"])

    if rag_type == "adaptive":
        return AdaptiveRAG(
            parts["llm"], parts["retriever"], parts["reranker"], parts["generator"]
        )

    raise ValueError(f"Unknown RAG type: {rag_type}. Choose from {RAG_TYPES}")
