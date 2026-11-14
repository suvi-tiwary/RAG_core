"""Loader for plain text and Markdown files."""

from rag_titan.loaders.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """Reads .txt and .md files as they are, so Markdown headings are kept."""

    extensions = [".txt", ".md", ".markdown"]

    def extract_text(self, path):
        """Return the file's text. Bytes that are not valid UTF-8 become "�" instead of failing."""
        with open(path, encoding="utf-8", errors="replace") as file:
            return file.read()
