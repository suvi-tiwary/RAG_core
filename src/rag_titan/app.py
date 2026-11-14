"""
RAGTitan: the easiest way to use the package.

It builds every part for you, so you only need two calls:

    from rag_titan import RAGTitan

    rag = RAGTitan(rag_type="simple")
    rag.index(["handbook.pdf", "policy.docx"])
    print(rag.ask("How many leave days do I get?"))

    print(rag.run("How many leave days do I get?")["log"])   # every step
    rag.evaluate("questions.json")                           # end to end, uses the LLM
    rag.evaluate_retrieval("questions.json")                 # search only, fast
    rag.evaluate_stages(files, "golden.json")                # every stage, scorecard
"""

from rag_titan.config import get_settings
from rag_titan.evaluation.benchmark import evaluate_retriever, format_report
from rag_titan.evaluation.collect import collect_dataset
from rag_titan.evaluation.end_to_end import evaluate_rag, format_rag_report
from rag_titan.evaluation.questions import as_questions
from rag_titan.evaluation.rag_evaluator import RAGEvaluator
from rag_titan.evaluation.reports.run import summary_text
from rag_titan.factory import build_rag, build_rag_parts
from rag_titan.rag_types.trace import describe_steps


class RAGTitan:
    """
    One object that can index files and answer questions about them.

    Args:
        rag_type: which RAG to use, "simple" or "adaptive".
        settings: optional dict of settings to change, for example
            {"llm_model": "llama3.2"}. Anything missing uses the
            value from the environment or the default.
    """

    def __init__(self, rag_type="simple", settings=None):
        self.settings = get_settings(settings)
        self.rag_type = rag_type

        # build all the parts once, they share one embedder and one store
        self.parts = build_rag_parts(self.settings)
        self.pipeline = self.parts["pipeline"]
        self.rag = build_rag(rag_type, self.parts)

    def index(self, file_paths, reset=True):
        """
        Read, chunk, embed and store the given files.

        Args:
            file_paths: list of paths to .pdf or .docx files.
            reset: when True, remove previously indexed chunks first.

        Returns:
            The number of chunks stored.
        """
        return self.pipeline.run(file_paths, reset=reset)

    def ask(self, question):
        """
        Answer a question using the indexed files.

        Args:
            question: the user's question as text.

        Returns:
            The answer text.
        """
        return self.rag.answer(question)

    def run(self, question):
        """
        Answer a question and show every step taken.

        Returns:
            The dict from the RAG type's run() ("answer", "sources", "steps"),
            plus "log": the steps as readable text (see describe_steps).
        """
        result = self.rag.run(question)
        result["log"] = describe_steps(result["steps"])
        return result

    def evaluate(self, questions, show=True):
        """
        Answer every test question with this RAG type and score the runs (uses the LLM).

        Args:
            questions: path to a questions JSON file, or a list of question dicts.
            show: print the report table.

        Returns:
            The dict from evaluate_rag(): evidence scores on the chunks given to
            the LLM, answered share, steps, time, and one row per question.
        """
        result = evaluate_rag(self.rag, questions, name=self.rag_type)
        if show:
            print(format_rag_report([result]))
        return result

    def evaluate_retrieval(self, questions, top_k=5, show=True):
        """
        Check how well search finds the evidence for test questions (no LLM used).

        Args:
            questions: path to a questions JSON file, or a list of question dicts.
            top_k: how many chunks to retrieve per question.
            show: print the report table.

        Returns:
            The dict from evaluate_retriever(): means, latency and one row per question.
        """
        result = evaluate_retriever(self.parts["retriever"], as_questions(questions), top_k)
        if show:
            print(format_report([result]))
        return result

    def evaluate_stages(self, file_paths, questions, k=5, include_embeddings=False, show=True):
        """
        Index the files, answer the golden questions and score every stage (uses the LLM).

        Args:
            file_paths: list of files to index (the index is rebuilt).
            questions: path to a golden questions JSON file, or a list of dicts
                with "id", "question" and gold labels (see collect_dataset).
            k: how many results to score in the ranking metrics.
            include_embeddings: also score the embedding stage.
            show: print the overall score, status and first failing stage.

        Returns:
            The report from RAGEvaluator: one entry per stage, "root_causes"
            and "scorecard". Save it with ReportManager.
        """
        dataset = collect_dataset(self, file_paths, questions, include_embeddings)
        report = RAGEvaluator(k=k).evaluate(**dataset)
        if show:
            print(summary_text(report))
        return report
