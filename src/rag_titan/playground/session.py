"""
What the browser playground calls: one object, plain JSON-ready dicts in and out.

    playground = Playground(json.loads(presets_text))
    playground.info()                                    # RAG types, FLOWs, questions, ...
    playground.run("paragraph", "adaptive", "q_002")     # answer, steps, scores
    playground.evaluate(["word", "paragraph"], ["simple", "adaptive"])
"""

from rag_titan.evaluation.end_to_end import evaluate_rag, format_rag_report
from rag_titan.factory import build_rag
from rag_titan.playground.presets import replay_parts


class Playground:
    """
    Runs the recorded presets: real code, recorded model replies and vectors.

    Args:
        presets: the dict saved by tools/playground/record_traces.py.
    """

    def __init__(self, presets):
        self.presets = presets
        self.parts = {}   # one index per chunking, built the first time it is used

    def info(self):
        """Everything the page needs to draw its controls and diagrams."""
        first = next(iter(self.presets["settings"]))
        return {
            "model": self.presets["model"],
            "embed_model": self.presets["embed_model"],
            "rag_types": list(self.presets["rag_types"]),
            "flows": {name: self._rag(first, name).FLOW for name in self.presets["rag_types"]},
            "chunkings": self.presets["settings"],
            "questions": self.presets["questions"],
            "documents": self.presets["documents"],
        }

    def run(self, chunking, rag_type, question_id):
        """
        Answer one preset question with one setup.

        Returns:
            A dict with "answer", "steps" (every step with input and output),
            "sources" (files given to the LLM), "recall", "answered" and "ms".
        """
        question = self._question(question_id)
        row = evaluate_rag(self._rag(chunking, rag_type), [question])["rows"][0]
        return {key: row[key] for key in
                ("answer", "steps", "sources", "recall", "ndcg", "answered", "ms")}

    def evaluate(self, chunkings, rag_types):
        """Every RAG type with every chunking on every preset question, like compare_setups()."""
        results = []
        for chunking in chunkings:
            for rag_type in rag_types:
                result = evaluate_rag(self._rag(chunking, rag_type), self.presets["questions"],
                                      name=f"{rag_type} + {chunking}")
                # the steps of every question are too much for a report: keep the scores
                result["rows"] = [{key: value for key, value in row.items() if key != "steps"}
                                  for row in result["rows"]]
                results.append(result)
        return {"results": results, "report": format_rag_report(results)}

    def _rag(self, chunking, rag_type):
        """Build a RAG type on the (cached) index of one chunking."""
        if chunking not in self.presets["settings"]:
            raise ValueError(f"Unknown chunking {chunking!r}: use one of "
                             f"{list(self.presets['settings'])}")
        if chunking not in self.parts:
            self.parts[chunking] = replay_parts(self.presets, chunking)
        return build_rag(rag_type, self.parts[chunking])

    def _question(self, question_id):
        """Return the preset question with this id."""
        for question in self.presets["questions"]:
            if question["id"] == question_id:
                return question
        raise ValueError(f"Unknown question id {question_id!r}")
