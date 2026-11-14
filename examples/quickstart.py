"""
Quick start: index your files and ask a question in a few lines.

Before running:
    pip install rag-titan
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run:
    python examples/quickstart.py path/to/file.pdf
"""

import sys

from rag_titan import RAGTitan, setup_logging


def main():
    """Index the files given on the command line and ask one question."""
    setup_logging("INFO")

    files = sys.argv[1:]
    if not files:
        print("Usage: python examples/quickstart.py file1.pdf [file2.docx ...]")
        return

    rag = RAGTitan(rag_type="simple")
    rag.index(files)

    print(rag.ask("What is this document about?"))


if __name__ == "__main__":
    main()
