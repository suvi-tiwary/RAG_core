"""Retrieval: find similar chunks, then rerank them."""

from rag_titan.retrieval.base_retriever import BaseRetriever
from rag_titan.retrieval.dedupe import overlap_fraction, remove_duplicates
from rag_titan.retrieval.dense_retriever import DenseRetriever
from rag_titan.retrieval.retriever import Retriever

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "Retriever",
    "overlap_fraction",
    "remove_duplicates",
]
