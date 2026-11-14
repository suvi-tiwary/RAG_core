"""
Which words of the source documents does a set of spans cover?

Counting covered words (not characters) is how the token metrics compare
retrieved chunks with gold evidence: Token Recall, Token Precision, Token IoU.
"""

from rag_titan.evaluation.stages.text import token_positions


def document_tokens(documents):
    """Return {doc_id: [(start, end), ...]}: the word offsets of every document."""
    return {doc_id: token_positions(text) for doc_id, text in documents.items()}


def covered_tokens(tokens, spans):
    """
    Return the set of (doc_id, word number) of every word a span touches.

    Args:
        tokens: the dict from document_tokens().
        spans: list of (doc_id, start, end).
    """
    covered = set()
    for doc_id, start, end in spans:
        for number, (word_start, word_end) in enumerate(tokens.get(doc_id, [])):
            if word_start < end and word_end > start:
                covered.add((doc_id, number))
    return covered
