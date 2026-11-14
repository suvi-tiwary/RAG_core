"""
Picks the right loader for a file based on its extension.
Add new loaders to the LOADERS list.
"""

import os

from rag_titan.loaders.docx_loader import DocxLoader
from rag_titan.loaders.pdf_loader import PdfLoader
from rag_titan.loaders.text_loader import TextLoader

LOADERS = [
    PdfLoader(),
    DocxLoader(),
    TextLoader(),
]


def get_loader(path):
    """
    Find the loader that can read this file.

    Raises:
        ValueError: when no loader supports the file extension.
    """
    extension = os.path.splitext(str(path))[1].lower()

    for loader in LOADERS:
        if extension in loader.extensions:
            return loader

    supported = [ext for loader in LOADERS for ext in loader.extensions]
    raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")


def load_document(path):
    """Load any supported file and return its document dict."""
    return get_loader(path).load(path)
