"""Tests for the sentence and paragraph splitters (handbook 3.1, 3.2)."""

import pytest

from rag_titan.chunking.structure_aware import split_paragraphs, split_sentences


def stripped(pieces):
    return [piece.strip() for piece in pieces]


@pytest.mark.parametrize("text, expected", [
    ("One. Two! Three? Four", ["One.", "Two!", "Three?", "Four"]),
    ("Dr. Smith is here. Mr. Lee too.", ["Dr. Smith is here.", "Mr. Lee too."]),
    ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
    ("It takes 3.5 days. Fine.", ["It takes 3.5 days.", "Fine."]),
    ("Steps: 1. Open it. 2. Sign it.", ["Steps: 1. Open it.", "2. Sign it."]),
    ("Use fruit, e.g. Apples. Or not.", ["Use fruit, e.g. Apples.", "Or not."]),
    ("The U.S. Army came. Then rain.", ["The U.S. Army came.", "Then rain."]),
    ("See Fig. a for details. Done.", ["See Fig. a for details.", "Done."]),
    ('She said "Hi." Then left.', ['She said "Hi."', "Then left."]),
    ("Wait... What? Ok.", ["Wait...", "What?", "Ok."]),
    ("東京は大きい。大阪も大きい。", ["東京は大きい。", "大阪も大きい。"]),
    ("Title\n\nBody text here", ["Title", "Body text here"]),
])
def test_split_sentences(text, expected):
    pieces = split_sentences(text)
    assert stripped(pieces) == expected
    assert "".join(pieces) == text


def test_split_paragraphs_keeps_every_character():
    text = "# Title\n\nFirst para.\nSame para.\n  \n\nSecond para."
    pieces = split_paragraphs(text)
    assert stripped(pieces) == ["# Title", "First para.\nSame para.", "Second para."]
    assert "".join(pieces) == text


def test_splitters_handle_empty_text():
    assert split_sentences("") == []
    assert split_paragraphs("") == []
