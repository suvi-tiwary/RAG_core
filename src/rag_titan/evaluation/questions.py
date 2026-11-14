"""
Load the test questions used to evaluate retrieval.

A questions file is a JSON list of records:
    [
      {"id": "q_001",
       "question": "How many leave days do I get?",
       "required_evidence_ids": ["hr.pdf#p4"],
       "task_type": "fact"}
    ]
"task_type" is optional. Other keys (like "answer" or "tenant") are kept as they are.
"""

import json
from pathlib import Path

from rag_titan.evaluation.metrics import parse_evidence_id


def load_questions(path):
    """
    Read and check a questions file.

    Args:
        path: path to the JSON file.

    Returns:
        The list of question dicts.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON (line {error.lineno}): {error.msg}")

    return check_questions(records)


def as_questions(questions):
    """
    Accept a questions file path or an already loaded list, and return the checked list.
    Lets every helper take either `"questions.json"` or `[{...}, ...]`.
    """
    if isinstance(questions, list):
        return check_questions(questions)
    return load_questions(questions)


def check_questions(records):
    """
    Check a list of question dicts and return it.
    Raises ValueError with the record id when something is missing or wrong.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("Questions must be a non-empty list of records")

    seen_ids = set()
    for number, record in enumerate(records, start=1):
        _check_record(record, number)
        if record["id"] in seen_ids:
            raise ValueError(f"Question id {record['id']!r} is used more than once")
        seen_ids.add(record["id"])

    return records


def _check_record(record, number):
    """Check one record; `number` (1, 2, ...) points to it in error messages."""
    if not isinstance(record, dict):
        raise ValueError(f"Question {number} must be a JSON object")

    name = record.get("id", f"number {number}")
    for key in ("id", "question"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Question {name!r}: {key!r} must be a non-empty text")

    evidence = record.get("required_evidence_ids")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"Question {name!r}: 'required_evidence_ids' must be a non-empty list")
    for label in evidence:
        try:
            parse_evidence_id(label)
        except ValueError as error:
            raise ValueError(f"Question {name!r}: {error}")

    task_type = record.get("task_type")
    if task_type is not None and not isinstance(task_type, str):
        raise ValueError(f"Question {name!r}: 'task_type' must be a text")
