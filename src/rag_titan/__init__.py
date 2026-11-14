"""
rag-titan: a Swiss Army knife for Retrieval-Augmented Generation (RAG).

Quick start:
    from rag_titan import RAGTitan

    rag = RAGTitan(rag_type="simple")
    rag.index(["handbook.pdf"])
    print(rag.ask("How many leave days do I get?"))
"""

from rag_titan.app import RAGTitan
from rag_titan.config import get_settings
from rag_titan.factory import RAG_TYPES, build_rag, build_rag_parts
from rag_titan.rag_types.adaptive.adaptive_rag import AdaptiveRAG
from rag_titan.rag_types.simple_rag import SimpleRAG
from rag_titan.utils.logger import setup_logging

__version__ = "0.8.5"

__all__ = [
    "RAGTitan",
    "SimpleRAG",
    "AdaptiveRAG",
    "RAG_TYPES",
    "build_rag",
    "build_rag_parts",
    "get_settings",
    "setup_logging",
    "__version__",
]
