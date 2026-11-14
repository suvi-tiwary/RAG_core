"""
Base class for every file loader.

To support a new file type, create a new loader, set its `extensions`,
implement extract_text(), and add it to loader_registry.py.
"""

import os


class BaseLoader:
    """Reads one kind of file and turns it into a document dict."""

    # file extensions this loader can read, for example [".pdf"]
    extensions = []

    def load(self, path):
        """
        Read a file and return a document dict.

        Args:
            path: path to the file.

        Returns:
            {"source": file name, "file_type": ".pdf", "text": all text}
        """
        path = str(path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        return {
            "source": os.path.basename(path),
            "file_type": os.path.splitext(path)[1].lower(),
            "text": self.extract_text(path),
        }

    def extract_text(self, path):
        """Return all text in the file as one string."""
        raise NotImplementedError
