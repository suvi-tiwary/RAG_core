"""
Golden template: a first draft of your golden questions, for a person to finish.

    from rag_titan.evaluation import make_golden_template

    records = make_golden_template(["How many leave days do I get?"],
                                   {"handbook.md": handbook_text})
    # records[0]["expected_evidence"] == ["Employees get 25 days of annual leave."]
    # records[0]["validated"] is False: a person must check it first

The suggested evidence is the sentence that shares the most words with the
question. It is only a suggestion: it can be the wrong sentence, so every
record starts with "validated": false and check_golden() says Ready: False
until a person has reviewed it.
"""

from rag_titan.chunking.structure_aware.sentences import split_sentences
from rag_titan.evaluation.stages.text import tokenize

TODO = "Check the suggested evidence, write expected_answer, then set validated to true"

# small words that say nothing about which sentence answers the question
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "be", "do", "does", "did", "i", "my",
              "me", "we", "you", "your", "can", "how", "what", "when", "where", "which",
              "who", "why", "of", "on", "in", "to", "for", "and", "or", "at", "by", "with",
              "it", "this", "that", "much", "many", "have", "has", "per"}


def make_golden_template(questions, documents=None):
    """
    Return one golden record per question, with empty labels to fill in.

    Args:
        questions: a list of question texts.
        documents: optional {name: text}; when given, each record gets a
            suggested evidence sentence taken word for word from them.

    Returns:
        A list of dicts with "id" (q1, q2, ...), "question", "expected_answer",
        "expected_evidence", "required_facts", "answerable", "validated" and "_todo".
    """
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a non-empty list of texts")
    if documents is not None and not isinstance(documents, dict):
        raise ValueError("documents must be a dict of {name: text}")

    records = []
    for number, question in enumerate(questions, start=1):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {number} is empty")
        suggestion = suggest_evidence(question, documents) if documents else None
        records.append({
            "id": f"q{number}",
            "question": question.strip(),
            "expected_answer": "",
            "expected_evidence": [suggestion] if suggestion else [],
            "required_facts": [],
            "answerable": True,
            "validated": False,
            "_todo": TODO,
        })
    return records


def suggest_evidence(question, documents):
    """
    Return the sentence of the documents that shares the most words with the
    question, or None when no sentence shares any word.

    Only sentences found exactly once are suggested, so check_golden()
    can find the quote again.

    Args:
        question: the question text.
        documents: {name: text}.
    """
    wanted = _key_words(question)
    best, best_score = None, 0
    for text in documents.values():
        for sentence in split_sentences(text):
            sentence = sentence.strip()
            score = len(wanted & _key_words(sentence))
            if score > best_score and _count(sentence, documents) == 1:
                best, best_score = sentence, score
    return best


def _key_words(text):
    """The words that carry meaning, with a plural "s" removed ("flights" -> "flight")."""
    words = set()
    for word in tokenize(text):
        if word in STOP_WORDS:
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def _count(sentence, documents):
    """How many times the sentence occurs in all documents together."""
    return sum(text.count(sentence) for text in documents.values())
