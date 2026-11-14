"""Loader for PDF files, using the pypdf library."""

from rag_titan.loaders.base_loader import BaseLoader

# form feed: the classic "next page" character, kept so chunkers can find pages
PAGE_BREAK = "\f"


class PdfLoader(BaseLoader):
    """Reads the text of every page in a .pdf file."""

    extensions = [".pdf"]

    def extract_text(self, path):
        """Return the text of all pages, separated by a page break ("\\f")."""
        # imported here, so the package also loads where pypdf is missing (the browser)
        from pypdf import PdfReader

        reader = PdfReader(path)

        # some pages (for example scanned images) have no text, so use ""
        pages = [page.extract_text() or "" for page in reader.pages]
        return PAGE_BREAK.join(pages)
