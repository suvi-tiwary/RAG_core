"""
Turn a raw metric value into a 0 to 1 score, where 1 is the best.

    direction        score                         example
    "high"           value                         recall 0.8          -> 0.80
    "low"            1 - value                     noise 0.1           -> 0.90
    "ratio"          1 / (1 + value)               redundancy 0.25     -> 0.80
    "fragmentation"  1 / max(1, value)             2 chunks per span   -> 0.50
    "cos_high"       (1 + cos) / 2                 cosine 0.6          -> 0.80
    "cos_low"        (1 - cos) / 2                 cosine 0            -> 0.50
    "cos_distance"   value / 2                     1 - cosine = 1      -> 0.50
    "margin"         clamp(0.5 + margin, 0, 1)     margin 0.25         -> 0.75
    "gain"           (1 + gain) / 2                no change           -> 0.50
    "target"         min(1, target / value)        2000 ms vs 1000 ms  -> 0.50
"""

from rag_titan.evaluation.stages.text import is_number

# direction -> (lowest valid value, highest valid value); None = no limit
RANGES = {
    "high": (0, 1), "low": (0, 1), "ratio": (0, None), "fragmentation": (0, None),
    "cos_high": (-1, 1), "cos_low": (-1, 1), "cos_distance": (0, 2), "margin": (-2, 2),
    "gain": (-1, 1), "target": (0, None),
}


def clamp(value):
    """Keep a score between 0 and 1."""
    return max(0.0, min(1.0, value))


def normalize(value, direction, target=None):
    """
    Return the 0 to 1 score of a raw metric value.

    Args:
        value: the measured value.
        direction: one of the names in the table above.
        target: for "target" only: the highest good value (e.g. 2000 ms), or
            a band {"min": 150, "max": 400}.
    """
    if direction not in RANGES:
        raise ValueError(f"Unknown direction {direction!r}")
    if not is_number(value):
        raise ValueError("value is not a finite number")

    low, high = RANGES[direction]
    if value < low or (high is not None and value > high):
        raise ValueError(f"value {value} is outside the range of a {direction!r} metric")

    if direction == "target":
        return target_score(value, target)
    formulas = {
        "high": value,
        "low": 1 - value,
        "ratio": 1 / (1 + value),
        "fragmentation": 1 / max(1, value),
        "cos_high": (1 + value) / 2,
        "cos_low": (1 - value) / 2,
        "cos_distance": value / 2,
        "margin": clamp(0.5 + value),      # +0.5 or more is a perfect separation
        "gain": (1 + value) / 2,
    }
    return formulas[direction]


def target_score(value, target):
    """
    Score a value against your own target.

    A number is a maximum: at or below it scores 1, twice as much scores 0.5.
    A {"min", "max"} band scores 1 inside, and less the further outside.
    """
    if isinstance(target, dict):
        low, high = target.get("min", 0), target.get("max")
        if not is_number(low) or not is_number(high) or low < 0 or high < low:
            raise ValueError("a target band needs 0 <= min <= max")
        if low <= value <= high:
            return 1.0
        return clamp(value / low) if value < low else clamp(high / value)

    if not is_number(target) or target <= 0:
        raise ValueError("a target must be a number above 0 or a {min, max} band")
    return 1.0 if value <= target else clamp(target / value)
