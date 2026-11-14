"""
Old import path, kept so code written for rag-titan 0.1.0 still works.
WordChunker now lives in rag_titan.chunking.fixed_size.word_chunker.
"""

from rag_titan.chunking.fixed_size.word_chunker import WordChunker

__all__ = ["WordChunker"]
