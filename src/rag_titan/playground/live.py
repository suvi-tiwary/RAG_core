"""
Live mode of the playground: your own models, your own question, your own text.

    live = LivePlayground(presets, chat=my_chat, embed=my_embed,
                          model="gemma3:4b", embed_model="nomic-embed-text")
    live.run("paragraph", "adaptive", "Can I work from Spain?")
    live.run("word", "simple", "What is the refund limit?",
             documents=[{"source": "my-notes.md", "text": "..."}])

In the browser, `chat` and `embed` call your local Ollama.
"""

import json

from rag_titan.embeddings.function_embedder import FunctionEmbedder
from rag_titan.evaluation.end_to_end import NOT_ANSWERED, evaluate_rag
from rag_titan.factory import build_rag
from rag_titan.llm.function_llm import FunctionLLM
from rag_titan.playground.presets import build_parts, index_documents


class LivePlayground:
    """
    Runs the RAG types with live models instead of recorded replies.

    Args:
        presets: the recorded presets (for the chunkings, sample documents and questions).
        chat: function messages -> reply text.
        embed: function texts -> one vector per text.
        model: chat model name, e.g. "gemma3:4b".
        embed_model: embedding model name, e.g. "nomic-embed-text".
    """

    def __init__(self, presets, chat, embed, model, embed_model):
        self.presets = presets
        self.llm = FunctionLLM(chat, model=model)
        self.embedder = FunctionEmbedder(embed, model=embed_model)
        self.parts = {}   # one index per (chunking, documents), embedded once

    def run(self, chunking, rag_type, question, documents=None):
        """
        Answer any question, on the sample documents or on `documents`.

        Returns:
            Like Playground.run(). "recall" and "ndcg" are None unless the
            question is one of the preset questions on the sample documents.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please enter a question")

        rag = build_rag(rag_type, self._parts(chunking, documents))
        preset = None if documents else self._preset_question(question)
        if preset:
            row = evaluate_rag(rag, [preset])["rows"][0]
            return {key: row[key] for key in
                    ("answer", "steps", "sources", "recall", "ndcg", "answered", "ms")}

        result = rag.run(question)
        return {"answer": result["answer"], "steps": result["steps"],
                "sources": [chunk["source"] for chunk in result["sources"]],
                "recall": None, "ndcg": None,
                "answered": result["answer"].strip() not in NOT_ANSWERED,
                "ms": sum(step["ms"] for step in result["steps"])}

    def _parts(self, chunking, documents):
        """Index the documents with one chunking, the first time they are used."""
        if chunking not in self.presets["settings"]:
            raise ValueError(f"Unknown chunking {chunking!r}")
        documents = _check_documents(documents) if documents else self.presets["documents"]

        key = (chunking, json.dumps(documents, sort_keys=True))
        if key not in self.parts:
            parts = build_parts(self.presets["settings"][chunking], self.embedder, self.llm)
            if index_documents(parts, documents) == 0:
                raise ValueError("The documents gave no chunks: add some text")
            self.parts[key] = parts
        return self.parts[key]

    def _preset_question(self, question):
        """The preset question with exactly this text, so its evidence can be scored."""
        for preset in self.presets["questions"]:
            if preset["question"].strip() == question.strip():
                return preset
        return None


def _check_documents(documents):
    """Documents must be a list of {"source": name, "text": text} dicts with some text."""
    if not isinstance(documents, list):
        raise ValueError("documents must be a list of {'source', 'text'} dicts")
    for document in documents:
        if not isinstance(document, dict) or not str(document.get("text", "")).strip():
            raise ValueError("Every document needs a 'source' name and some 'text'")
        document.setdefault("source", "your-text.md")
    return documents
