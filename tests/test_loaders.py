"""Tests for the loaders keeping document structure (paragraphs, headings, pages)."""

from docx import Document

from rag_titan.chunking import WordChunker
from rag_titan.loaders import DocxLoader, PdfLoader, TextLoader, get_loader
from rag_titan.loaders.docx_loader import heading_level


def make_docx(path):
    document = Document()
    document.add_heading("Leave policy", level=1)
    document.add_paragraph("Staff get twenty days.")
    document.add_paragraph("   ")  # blank paragraphs are skipped
    document.add_heading("Sick leave", level=2)
    document.add_paragraph("Tell your manager.")
    document.save(path)


def test_docx_keeps_paragraphs_and_headings(tmp_path):
    path = tmp_path / "policy.docx"
    make_docx(path)

    text = DocxLoader().load(path)["text"]

    assert text == "# Leave policy\n\nStaff get twenty days.\n\n## Sick leave\n\nTell your manager."


def test_heading_level_reads_word_styles():
    assert heading_level("Title") == 1
    assert heading_level("Heading 3") == 3
    assert heading_level("Heading 9") == 6
    assert heading_level("Normal") == 0
    assert heading_level("Heading x") == 0


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


def test_pdf_separates_pages_with_page_break(monkeypatch, tmp_path):
    path = tmp_path / "handbook.pdf"
    path.write_bytes(b"%PDF fake")
    pages = [FakePage("Page one."), FakePage(None), FakePage("Three.")]
    fake_reader = type("FakeReader", (), {"pages": pages})
    # pypdf is imported when a file is read, so patch it in the pypdf module
    monkeypatch.setattr("pypdf.PdfReader", lambda _path: fake_reader)

    assert PdfLoader().load(path)["text"] == "Page one.\f\fThree."


def test_text_loader_reads_markdown_and_txt(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nBody with é.", encoding="utf-8")

    document = get_loader(path).load(path)

    assert isinstance(get_loader("a.txt"), TextLoader)
    assert document == {"source": "notes.md", "file_type": ".md", "text": "# Title\n\nBody with é."}


def test_text_loader_survives_bad_bytes(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_bytes(b"ok \xff\xfe end")
    assert TextLoader().load(path)["text"].startswith("ok ")


def test_word_chunker_ignores_the_new_separators():
    # loaders now use "\n\n" and "\f" where they used spaces; word chunks stay the same
    old = "Page one. Staff get twenty days. Three."
    new = "Page one.\fStaff get\n\ntwenty days.\fThree."
    chunker = WordChunker(chunk_size=3, overlap=1)
    assert chunker.split_text(new) == chunker.split_text(old)
