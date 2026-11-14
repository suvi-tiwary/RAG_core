"""
Split text into paragraphs and sentences without losing a single character.

No libraries: simple rules that handle the common traps (handbook 3.1):
abbreviations (Dr., e.g.), initials (J. K.), list numbers (1.), decimals (3.5).
Every piece keeps the spaces or newlines after it, so "".join(pieces) == text.
"""

import re

# a blank line (spaces allowed on it) ends a paragraph
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")

# end punctuation + closing quotes/brackets + space; CJK and Hindi marks need no space
SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+|[。！？।]+[\"'”’」』)）\]]*\s*")

# a "." after these words does not end a sentence
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig",
    "approx", "inc", "ltd", "dept", "est", "no", "vol", "p", "pp", "cf",
}


def cut_after(text, ends):
    """Cut `text` after each position in `ends`; joining the pieces gives `text`."""
    pieces, start = [], 0

    for end in ends:
        if end > start:
            pieces.append(text[start:end])
            start = end

    if start < len(text):
        pieces.append(text[start:])
    return pieces


def split_paragraphs(text):
    """Cut `text` after every blank line."""
    return cut_after(text, [match.end() for match in PARAGRAPH_BREAK.finditer(text)])


def word_before(text, end):
    """The word just before position `end`, without brackets or quotes in front."""
    match = re.search(r"[^\s(\[\"'“‘]+$", text[:end])
    return match.group().lower() if match else ""


def is_sentence_end(text, match):
    """Decide if a punctuation match really ends a sentence."""
    after = text[match.end():match.end() + 1]
    if after.islower():
        return False  # "Fig. a" or "etc. and" continue the sentence

    if match.group()[0] == "." and not match.group().startswith(".."):
        word = word_before(text, match.start())
        if word in ABBREVIATIONS or word.isdigit():
            return False  # "Dr. Smith", "1. Open the form"
        if re.fullmatch(r"[a-z]|(?:[a-z]\.)+[a-z]", word):
            return False  # initials "J. K." and dotted forms "e.g.", "U.S."

    return True


def split_sentences(text):
    """Cut `text` into sentences; a blank line always ends a sentence too."""
    sentences = []

    for paragraph in split_paragraphs(text):
        ends = [m.end() for m in SENTENCE_END.finditer(paragraph) if is_sentence_end(paragraph, m)]
        sentences.extend(cut_after(paragraph, ends))

    return sentences
