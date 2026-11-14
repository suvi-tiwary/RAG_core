"""
Evaluation: measure how well a RAG system works.

Two ways to use it:
    evaluate_rag(), compare_setups(), evaluate_retriever()
        run rag-titan pipelines on test questions and score them.
    RAGEvaluator and the stage evaluators (ChunkEvaluator, RetrievalEvaluator, ...)
        score the logged outputs of any RAG system: data in, report out.
"""

from rag_titan.evaluation.benchmark import compare_retrievers, evaluate_retriever, format_report
from rag_titan.evaluation.collect import collect_dataset
from rag_titan.evaluation.dataset import build_dataset
from rag_titan.evaluation.end_to_end import compare_setups, evaluate_rag, format_rag_report
from rag_titan.evaluation.golden import check_golden, load_golden
from rag_titan.evaluation.golden_template import make_golden_template
from rag_titan.evaluation.metrics import (
    matches_evidence,
    ndcg_at_k,
    parse_evidence_id,
    recall_at_k,
    reciprocal_rank,
)
from rag_titan.evaluation.profiling import profile_documents
from rag_titan.evaluation.questions import as_questions, check_questions, load_questions
from rag_titan.evaluation.rag_evaluator import RAGEvaluator
from rag_titan.evaluation.reports import ReportManager, evaluate_file
from rag_titan.evaluation.root_cause import RootCauseAnalyzer
from rag_titan.evaluation.sampling import sample_documents
from rag_titan.evaluation.scoring import ScoreAggregator
from rag_titan.evaluation.stages import (
    AugmentationEvaluator,
    ChunkEvaluator,
    EmbeddingEvaluator,
    EndToEndEvaluator,
    GenerationEvaluator,
    RerankingEvaluator,
    RetrievalEvaluator,
)

__all__ = [
    "AugmentationEvaluator",
    "ChunkEvaluator",
    "EmbeddingEvaluator",
    "EndToEndEvaluator",
    "GenerationEvaluator",
    "RAGEvaluator",
    "ReportManager",
    "RerankingEvaluator",
    "RetrievalEvaluator",
    "RootCauseAnalyzer",
    "ScoreAggregator",
    "as_questions",
    "build_dataset",
    "check_golden",
    "collect_dataset",
    "compare_setups",
    "evaluate_file",
    "evaluate_rag",
    "format_rag_report",
    "check_questions",
    "compare_retrievers",
    "evaluate_retriever",
    "format_report",
    "load_golden",
    "load_questions",
    "make_golden_template",
    "matches_evidence",
    "ndcg_at_k",
    "parse_evidence_id",
    "profile_documents",
    "recall_at_k",
    "reciprocal_rank",
    "sample_documents",
]
