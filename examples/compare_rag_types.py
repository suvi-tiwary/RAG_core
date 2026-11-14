"""
Ask the same question with every RAG type and compare the answers.

Run:
    python examples/compare_rag_types.py path/to/file.pdf "your question"
"""

import sys

from rag_titan import RAG_TYPES, build_rag, build_rag_parts, get_settings


def main():
    """Index once, then answer with each RAG type."""
    if len(sys.argv) < 3:
        print('Usage: python examples/compare_rag_types.py file.pdf "question"')
        return

    file_path, question = sys.argv[1], sys.argv[2]

    # build the parts once so every RAG type shares the same index
    parts = build_rag_parts(get_settings())
    parts["pipeline"].run([file_path])

    for rag_type in RAG_TYPES:
        rag = build_rag(rag_type, parts)
        print(f"\n=== {rag_type} ===")
        print(rag.answer(question))


if __name__ == "__main__":
    main()
