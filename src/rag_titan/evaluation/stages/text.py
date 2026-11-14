"""
Small text and number helpers shared by every stage evaluator.

A "token" here is a word: letters, digits and underscores. The same rule is
used everywhere, so scores from different stages can be compared.
"""

import math
import re

WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Return the lowercase words of a text, e.g. "Hi, Bob!" -> ["hi", "bob"]."""
    return WORD.findall((text or "").lower())


def token_positions(text):
    """Return the (start, end) character offsets of every word in the text."""
    return [(match.start(), match.end()) for match in WORD.finditer(text or "")]


def average(values):
    """
    Mean of the values that were measured (not None).
    Returns None when nothing was measured, so "no data" never looks like 0.
    """
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)


def percentile(values, percent=95):
    """
    The value below which `percent` % of the measured values fall.
    Uses linear interpolation between the two nearest values.

    Args:
        values: numbers; None entries are ignored.
        percent: 0 to 100, e.g. 50 for the median and 95 for slow requests.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be between 0 and 100, got {percent!r}")

    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None

    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cosine(first, second):
    """
    Cosine similarity of two vectors (-1 to 1).
    Returns None when it cannot be measured: an empty or all-zero vector.
    """
    if len(first) != len(second):
        raise ValueError(f"Vectors have different sizes: {len(first)} and {len(second)}")

    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    if not norm_first or not norm_second:
        return None

    dot = sum(a * b for a, b in zip(first, second))
    return dot / (norm_first * norm_second)


def jaccard(first, second):
    """
    Share of words two texts have in common (0 to 1): |A ∩ B| / |A ∪ B|.
    Returns None when both texts have no words.
    """
    words_first = set(tokenize(first))
    words_second = set(tokenize(second))
    if not words_first | words_second:
        return None
    return len(words_first & words_second) / len(words_first | words_second)


def is_number(value):
    """Return True for a real, finite number (True/False do not count)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)
