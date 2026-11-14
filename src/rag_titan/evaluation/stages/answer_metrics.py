"""
Answer metrics: compare a generated answer with a reference answer, without an LLM judge.

Notation: A is the generated answer, Y the reference answer, t(A) and t(Y)
their lowercase word lists. Texts are compared after this normalisation, so
"Hybrid search." equals "hybrid search".
"""

import re
from collections import Counter

from rag_titan.evaluation.stages.text import tokenize

# 25, 3.5 and 1,000 are numbers; commas are dropped before comparing
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def exact_match(answer, expected):
    """
    1.0 when the answer has exactly the words of the reference, else 0.0.

        EM = 1[norm(A) = norm(Y)]
    """
    return float(tokenize(answer) == tokenize(expected))


def f1_of(precision, recall):
    """Harmonic mean of precision and recall: 2 P R / (P + R)."""
    if not precision + recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_scores(answer, expected):
    """
    Word overlap of the answer and the reference (order does not matter).

        Answer Token Precision P_a = #matched word occurrences / |t(A)|
        Answer Token Recall    R_a = #matched word occurrences / |t(Y)|
        Answer F1                  = 2 P_a R_a / (P_a + R_a)
    """
    answer_words, expected_words = tokenize(answer), tokenize(expected)
    if not answer_words or not expected_words:
        same = float(answer_words == expected_words)
        return {"answer_precision": same, "answer_recall": same, "answer_f1": same}

    matched = sum((Counter(answer_words) & Counter(expected_words)).values())
    precision = matched / len(answer_words)
    recall = matched / len(expected_words)
    return {"answer_precision": precision, "answer_recall": recall,
            "answer_f1": f1_of(precision, recall)}


def lcs_length(first, second):
    """Length of the longest common subsequence: shared words in the same order."""
    row = [0] * (len(second) + 1)
    for word in first:
        previous = 0   # the value diagonally up-left of the cell being filled
        for column, other in enumerate(second, start=1):
            above = row[column]
            row[column] = previous + 1 if word == other else max(above, row[column - 1])
            previous = above
    return row[-1]


def rouge_l_f1(answer, expected):
    """
    ROUGE-L F1: like Answer F1, but the shared words must keep their order.

        P_LCS = LCS(A, Y) / |t(A)|,  R_LCS = LCS(A, Y) / |t(Y)|
        ROUGE-L F1 = 2 P_LCS R_LCS / (P_LCS + R_LCS)
    """
    answer_words, expected_words = tokenize(answer), tokenize(expected)
    if not answer_words or not expected_words:
        return float(answer_words == expected_words)

    shared = lcs_length(answer_words, expected_words)
    return f1_of(shared / len(answer_words), shared / len(expected_words))


def required_fact_coverage(answer, facts):
    """
    Share of the required facts written in the answer (case does not matter).

        Required Fact Coverage = #required facts found in the answer / #required facts
    """
    if not facts:
        return None
    lowered = answer.lower()
    return sum(fact.lower() in lowered for fact in facts) / len(facts)


def citation_scores(citations, expected):
    """
    Compare the cited source ids with the expected ones.

        Citation Precision = #correct cited ids / #cited ids
        Citation Recall    = #correct cited ids / #expected ids
    """
    scores = {"citation_precision": None, "citation_recall": None, "citation_f1": None}
    if citations is None or expected is None:
        return scores

    cited, gold = set(citations), set(expected)
    if cited:
        scores["citation_precision"] = len(cited & gold) / len(cited)
    if gold:
        scores["citation_recall"] = len(cited & gold) / len(gold)
    if cited and gold:
        scores["citation_f1"] = f1_of(scores["citation_precision"], scores["citation_recall"])
    return scores


def context_grounding(answer, context):
    """
    Share of the answer's words that also occur in the context given to the LLM.
    A low value is a hint (not a proof) that the answer was not taken from the context.
    """
    words = tokenize(answer)
    if not words or context is None:
        return None
    known = set(tokenize(context))
    return sum(word in known for word in words) / len(words)


def unsupported_number_rate(answer, context):
    """
    Share of the numbers in the answer that are not in the context (lower is better).
    Invented numbers are a common, easy to detect kind of hallucination.
    """
    if context is None:
        return None
    numbers = [number.replace(",", "") for number in NUMBER.findall(answer)]
    if not numbers:
        return None
    known = {number.replace(",", "") for number in NUMBER.findall(context)}
    return sum(number not in known for number in numbers) / len(numbers)
