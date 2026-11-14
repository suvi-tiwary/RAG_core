"""File loaders that turn files into plain text documents."""

from rag_titan.loaders.base_loader import BaseLoader
from rag_titan.loaders.docx_loader import DocxLoader
from rag_titan.loaders.loader_registry import get_loader, load_document
from rag_titan.loaders.pdf_loader import PdfLoader
from rag_titan.loaders.text_loader import TextLoader

__all__ = [
    "BaseLoader",
    "DocxLoader",
    "PdfLoader",
    "TextLoader",
    "get_loader",
    "load_document",
]
