"""Loader for Word files, using the python-docx library."""

from rag_titan.loaders.base_loader import BaseLoader


def heading_level(style_name):
    """Return 1-6 for Word heading styles ("Title", "Heading 2", ...), else 0."""
    if style_name == "Title":
        return 1

    if style_name.startswith("Heading "):
        level = style_name.split(" ", 1)[1]
        if level.isdigit():
            return min(int(level), 6)

    return 0


class DocxLoader(BaseLoader):
    """
    Reads the text of every paragraph in a .docx file.

    Paragraphs are separated by a blank line, and Word headings become
    Markdown headings ("Heading 2" -> "## "), so chunkers can see the
    document's structure.
    """

    extensions = [".docx"]

    def extract_text(self, path):
        """Return all non-empty paragraphs, separated by blank lines."""
        # imported here, so the package also loads where python-docx is missing (the browser)
        from docx import Document

        blocks = []
        for para in Document(path).paragraphs:
            text = para.text.strip()
            if not text:
                continue

            level = heading_level(para.style.name if para.style is not None else "")
            blocks.append("#" * level + " " + text if level else text)

        return "\n\n".join(blocks)
