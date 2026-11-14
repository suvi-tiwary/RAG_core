"""
A tiny stand-in for YOUR RAG, so you can try the whole guide without a model.

    python my_rag.py              # version 1
    python my_rag.py --improved   # version 2: ignores filler words, can say "I don't know"

It answers every line of questions.txt and saves what it did in my_rag_logs.json.
HashEmbedder plays "your embedding model" and picking a sentence plays "your LLM".
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from rag_titan.embeddings import HashEmbedder

IMPROVED = "--improved" in sys.argv
TOP_K = 3                               # chunks returned by search
MIN_SCORE = 0.4                         # version 2 refuses below this similarity
FILLER = set("a an the is are be by of for to on in i can how much what who has have "
             "there must also your".split())
model = HashEmbedder(dimensions=64)    # <- your embedding model


def key_words(text):
    """Lower-case words; version 2 drops filler words and a plural "s"."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not IMPROVED:
        return words
    return [word.rstrip("s") for word in words if word not in FILLER]


def embed(texts):
    """Return one unit-length vector per text."""
    vectors = model.embed_many([" ".join(key_words(text)) for text in texts])
    return [vector / (np.linalg.norm(vector) or 1.0) for vector in vectors]


def load_chunks():
    """Indexing: one chunk per paragraph, with the metadata your index stores."""
    text = Path("documents/travel_policy.md").read_text(encoding="utf-8")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return [{"id": f"c{number}", "text": paragraph,
             "metadata": {"source": "travel_policy.md", "page": 1}}
            for number, paragraph in enumerate(paragraphs, start=1)]


def write_answer(question, chunk):
    """Your LLM: here, the sentence of the best chunk that shares most words with the question."""
    asked = set(key_words(question))
    sentences = re.split(r"(?<=\.) ", chunk["text"])
    best = max(sentences, key=lambda sentence: len(asked & set(key_words(sentence))))
    return f'{best} [{chunk["id"]}]'


def answer_question(question, chunks, vectors):
    """Search, build the context, write the answer. Returns everything that happened."""
    query_vector = embed([question])[0]
    scores = [float(query_vector @ vector) for vector in vectors]
    ranked = sorted(range(len(chunks)), key=lambda index: -scores[index])[:TOP_K]
    hits = [chunks[index]["id"] for index in ranked]

    if IMPROVED and scores[ranked[0]] < MIN_SCORE:
        answer = "I don't know based on the documents."
    else:
        answer = write_answer(question, chunks[ranked[0]])
    return {"question": question, "hits": hits, "prompt_chunks": hits[:2], "answer": answer,
            "query_vector": query_vector.round(3).tolist(),
            "search_ms": 12, "llm_ms": 840}      # fixed here; your RAG measures them


def main():
    chunks = load_chunks()
    vectors = embed([chunk["text"] for chunk in chunks])
    questions = Path("questions.txt").read_text(encoding="utf-8").splitlines()
    logs = {"chunks": chunks,
            "vectors": {chunk["id"]: vector.round(3).tolist()
                        for chunk, vector in zip(chunks, vectors)},
            # one vector per sentence: only for the "semantic cohesion" chunking metrics
            "sentence_vectors": {chunk["id"]: [vector.round(3).tolist() for vector in
                                               embed(re.split(r"(?<=\.) ", chunk["text"]))]
                                 for chunk in chunks},
            "runs": [answer_question(question, chunks, vectors) for question in questions]}
    Path("my_rag_logs.json").write_text(json.dumps(logs, indent=1), encoding="utf-8")
    print(f"Answered {len(questions)} questions -> my_rag_logs.json")


if __name__ == "__main__":
    main()
