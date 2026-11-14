"""
Playground presets: real model runs, recorded once, replayed anywhere.

    presets = record_presets(documents, questions, PRESET_CHUNKING, llm, embedder)
    problems = check_presets(presets)       # [] when a replay gives the same runs

The recorder (tools/playground/record_traces.py), the tests and the browser all
build their RAG types with build_parts(), so a replay is the same code as the recording.
"""

from rag_titan.config import get_settings
from rag_titan.embeddings.replay_embedder import RecordingEmbedder, ReplayEmbedder
from rag_titan.factory import RAG_TYPES, build_chunker, build_generator, build_rag, build_reranker
from rag_titan.llm.replay_llm import RecordingLLM, ReplayLLM
from rag_titan.retrieval.dense_retriever import DenseRetriever
from rag_titan.vector_stores.memory_store import MemoryStore

# the ingestion choices offered in the playground
PRESET_CHUNKING = {
    "word": {"chunker": "word", "chunk_size": 40, "chunk_overlap": 20},
    "sentence": {"chunker": "sentence", "chunk_size": 300, "chunk_overlap": 0},
    "paragraph": {"chunker": "paragraph", "chunk_size": 300, "chunk_overlap": 0},
}

# the only settings a preset needs; saved with it so every machine uses the same ones
PRESET_KEYS = ["chunker", "chunk_size", "chunk_overlap", "chunk_encoding",
               "dedupe_overlap", "reranker", "generator"]


def preset_settings(overrides):
    """The settings of one preset: the defaults plus `overrides`, only PRESET_KEYS."""
    settings = get_settings(overrides)
    return {key: settings[key] for key in PRESET_KEYS}


def build_parts(settings, embedder, llm):
    """Build every part of a RAG from preset settings, a model and an embedder."""
    store = MemoryStore()
    return {
        "llm": llm,
        "chunker": build_chunker(settings),
        "embedder": embedder,
        "store": store,
        "retriever": DenseRetriever(embedder, store, dedupe_overlap=settings["dedupe_overlap"]),
        "reranker": build_reranker(settings, llm),
        "generator": build_generator(settings, llm),
    }


def index_documents(parts, documents):
    """Chunk, embed and store documents given as {"source", "text"} dicts (no files needed)."""
    chunks = parts["chunker"].chunk(documents)
    vectors = parts["embedder"].embed_many([chunk["text"] for chunk in chunks])

    parts["store"].clear()
    parts["store"].add(chunks, vectors)
    return len(chunks)


def summarize_run(result):
    """The part of a run() result a replay must repeat exactly (timings change)."""
    return {
        "answer": result["answer"],
        "steps": [step["step"] for step in result["steps"]],
        "sources": [chunk["id"] for chunk in result["sources"]],
    }


def record_presets(documents, questions, chunking, llm, embedder):
    """
    Run every RAG type with every chunking on every question, and record it.

    Args:
        documents: list of {"source", "text"} dicts.
        questions: list of question dicts (see load_questions).
        chunking: dict of {name: setting overrides}, e.g. PRESET_CHUNKING.
        llm: the real model, e.g. OllamaLLM("gemma3:4b").
        embedder: the real embedder, e.g. OllamaEmbedder("nomic-embed-text").

    Returns:
        A JSON-ready dict: documents, questions, settings, the recorded replies and
        vectors, and the expected summary of every run.
    """
    llm, embedder = RecordingLLM(llm), RecordingEmbedder(embedder)
    presets = {"model": llm.model, "embed_model": embedder.model, "documents": documents,
               "questions": questions, "rag_types": list(RAG_TYPES),
               "settings": {name: preset_settings(o) for name, o in chunking.items()},
               "expected": {}}

    for name, settings in presets["settings"].items():
        parts = build_parts(settings, embedder, llm)
        index_documents(parts, documents)
        presets["expected"][name] = _run_all(parts, questions)

    presets["replies"], presets["vectors"] = llm.replies, embedder.vectors
    return presets


def replay_parts(presets, name):
    """Build and index the parts of one preset chunking, answering from the recording."""
    llm = ReplayLLM(presets["replies"], model=presets["model"])
    embedder = ReplayEmbedder(presets["vectors"], model=presets["embed_model"])

    parts = build_parts(presets["settings"][name], embedder, llm)
    index_documents(parts, presets["documents"])
    return parts


def check_presets(presets):
    """Replay every recorded run; return a list of differences ([] means all the same)."""
    problems = []

    for name, expected in presets["expected"].items():
        replayed = _run_all(replay_parts(presets, name), presets["questions"])
        for rag_type, runs in expected.items():
            for question_id, run in runs.items():
                again = replayed.get(rag_type, {}).get(question_id)
                if again != run:
                    problems.append(f"{name} / {rag_type} / {question_id}: "
                                    f"recorded {run}, replayed {again}")

    return problems


def _run_all(parts, questions):
    """{rag_type: {question id: summarize_run(...)}} for every RAG type and question."""
    runs = {}
    for rag_type in RAG_TYPES:
        rag = build_rag(rag_type, parts)
        runs[rag_type] = {q["id"]: summarize_run(rag.run(q["question"])) for q in questions}
    return runs
