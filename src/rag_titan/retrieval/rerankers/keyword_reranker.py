"""Reranker that rewards chunks sharing words with the question."""

from rag_titan.retrieval.rerankers.base_reranker import BaseReranker

# table used to remove punctuation before comparing words
PUNCTUATION = str.maketrans("", "", ".,?!:;()\"'")


class KeywordReranker(BaseReranker):
    """
    Adds a small bonus for every word shared by the question and the chunk.
    final score = semantic score + keyword_weight * shared words

    Args:
        keyword_weight: bonus added for each shared word.
    """

    def __init__(self, keyword_weight=0.05):
        self.keyword_weight = keyword_weight

    def score(self, query, match):
        """Return the semantic score plus the keyword bonus."""
        shared_words = self._words(query) & self._words(match["text"])
        return match["score"] + self.keyword_weight * len(shared_words)

    def _words(self, text):
        """Return the set of lowercase words in the text."""
        return set(text.lower().translate(PUNCTUATION).split())
