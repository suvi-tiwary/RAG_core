"""
Old home of the retriever, kept so code written for 0.5.x keeps working.

`Retriever` is now `DenseRetriever` (see dense_retriever.py).
"""

from rag_titan.retrieval.dense_retriever import DenseRetriever

Retriever = DenseRetriever

__all__ = ["Retriever"]
