"""
Tests for running without ollama, pypdf, python-docx and python-dotenv (like in
the browser), HashEmbedder, and recording/replaying models.
The Ollama host settings are tested in test_ollama_models.py.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from conftest import FakeEmbedder, FakeLLM

from rag_titan import SimpleRAG, get_settings
from rag_titan.embeddings import HashEmbedder, RecordingEmbedder, ReplayEmbedder
from rag_titan.factory import build_embedder
from rag_titan.generation import GroundedGenerator
from rag_titan.llm import RecordingLLM, ReplayLLM

BLOCKED_RUN = textwrap.dedent("""
    import sys
    for name in ["ollama", "pypdf", "docx", "dotenv"]:
        sys.modules[name] = None          # makes `import name` fail, like in Pyodide

    from rag_titan import SimpleRAG, get_settings
    from rag_titan.chunking import WordChunker
    from rag_titan.embeddings import HashEmbedder
    from rag_titan.generation import GroundedGenerator
    from rag_titan.llm import BaseLLM
    from rag_titan.pipelines import IndexingPipeline
    from rag_titan.retrieval import DenseRetriever
    from rag_titan.retrieval.rerankers import KeywordReranker
    from rag_titan.vector_stores import MemoryStore

    class EchoLLM(BaseLLM):
        def chat(self, messages, tools=None):
            return {"content": "20 days", "tool_calls": [], "message": {}}

    assert get_settings()["chunker"] == "word"           # works without dotenv
    embedder, store = HashEmbedder(), MemoryStore()
    IndexingPipeline(WordChunker(8, 0), embedder, store).run([sys.argv[1]])
    rag = SimpleRAG(DenseRetriever(embedder, store), KeywordReranker(),
                    GroundedGenerator(EchoLLM("echo")))
    result = rag.run("how many leave days")
    print(result["answer"], result["sources"][0]["source"])
""")


def test_package_runs_without_ollama_pypdf_docx_and_dotenv(tmp_path):
    (tmp_path / "hr.md").write_text("You get 20 leave days. Salary is paid monthly.",
                                    encoding="utf-8")
    script = tmp_path / "blocked.py"
    script.write_text(BLOCKED_RUN, encoding="utf-8")

    done = subprocess.run([sys.executable, str(script), str(tmp_path / "hr.md")],
                          capture_output=True, text=True, timeout=60)

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "20 days hr.md"


def test_hash_embedder_is_deterministic_and_finds_shared_words():
    embedder = HashEmbedder(dimensions=256)
    leave, salary, query = embedder.embed_many(
        ["you get 20 leave days", "salary is paid monthly", "how many leave days"])

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert np.array_equal(HashEmbedder(dimensions=256).embed("you get 20 leave days"), leave)
    assert leave.shape == (256,) and leave.dtype == np.float32
    assert cosine(query, leave) > cosine(query, salary)


@pytest.mark.parametrize("bad", [0, 4, 7.5, "512"])
def test_hash_embedder_bad_dimensions(bad):
    with pytest.raises(ValueError, match="dimensions"):
        HashEmbedder(dimensions=bad)


def test_hash_embedder_from_settings():
    assert isinstance(build_embedder(get_settings({"embed_provider": "hash"})), HashEmbedder)


def test_record_then_replay_gives_the_same_run(parts):
    recorder = RecordingLLM(FakeLLM(["20 leave days [chunk_0]"]))
    first = SimpleRAG(parts["retriever"], parts["reranker"],
                      GroundedGenerator(recorder)).run("how many leave days?")

    replay = ReplayLLM(recorder.replies, model="fake-model")
    second = SimpleRAG(parts["retriever"], parts["reranker"],
                       GroundedGenerator(replay)).run("how many leave days?")

    assert len(recorder.replies) == 1
    assert second["answer"] == first["answer"] == "20 leave days [chunk_0]"
    assert [s["output"] for s in second["steps"]] == [s["output"] for s in first["steps"]]


def test_replay_llm_refuses_a_prompt_it_never_saw(parts):
    rag = SimpleRAG(parts["retriever"], parts["reranker"], GroundedGenerator(ReplayLLM({})))
    with pytest.raises(ValueError, match="No recorded reply"):
        rag.run("how many leave days?")
    with pytest.raises(ValueError, match="dict"):
        ReplayLLM(["not", "a", "dict"])
    with pytest.raises(ValueError, match="tool calls"):
        RecordingLLM(FakeLLM()).chat([{"role": "user", "content": "hi"}], tools=[print])


def test_record_then_replay_embedder():
    recorder = RecordingEmbedder(FakeEmbedder(), decimals=3)
    recorded = recorder.embed_many(["leave days", "remote work"])
    replayed = ReplayEmbedder(recorder.vectors).embed_many(["remote work", "leave days"])

    assert np.array_equal(replayed[0], recorded[1]) and np.array_equal(replayed[1], recorded[0])
    with pytest.raises(ValueError, match="No recorded vector"):
        ReplayEmbedder(recorder.vectors).embed("salary")
    with pytest.raises(ValueError, match="dict"):
        ReplayEmbedder([])

