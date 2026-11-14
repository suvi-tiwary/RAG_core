"""Chunker that splits text by counting model tokens."""

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size.fixed_windows import check_window_size, fixed_windows

INSTALL_HINT = (
    "TokenChunker uses tiktoken by default. Install it with "
    '`pip install "rag-titan[tokens]"`, or pass your own tokenizer=...'
)


def load_tiktoken(encoding):
    """
    Return the tiktoken tokenizer called `encoding`.

    Args:
        encoding: a tiktoken encoding name, for example "cl100k_base".
    """
    try:
        import tiktoken
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error

    return tiktoken.get_encoding(encoding)


class TokenChunker(BaseChunker):
    """
    Splits text into fixed-size windows of model tokens that overlap.
    Use the same tokenizer as your embedding model, so no chunk is ever
    longer than the model accepts.
    Example: size 256, overlap 32 -> each new chunk starts 224 tokens later.

    Args:
        chunk_size: number of tokens in each chunk.
        overlap: number of tokens shared by two neighbouring chunks.
        encoding: tiktoken encoding name, used when no tokenizer is given.
        tokenizer: your own tokenizer, any object with encode(text) -> ids
            and decode(ids) -> text. For a Hugging Face tokenizer, wrap it so
            encode() calls encode(text, add_special_tokens=False).
    """

    def __init__(self, chunk_size=256, overlap=32, encoding="cl100k_base", tokenizer=None):
        check_window_size(chunk_size, overlap)

        if tokenizer is None:
            tokenizer = load_tiktoken(encoding)
        elif not (hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode")):
            raise ValueError("tokenizer must have encode(text) and decode(ids) methods")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer

    def split_text(self, text):
        """Return overlapping pieces of `chunk_size` tokens each."""
        # blank text has nothing worth indexing, so it gives no chunks
        if not text.strip():
            return []

        ids = list(self.tokenizer.encode(text))
        windows = fixed_windows(ids, self.chunk_size, self.overlap)
        return [self.tokenizer.decode(window) for window in windows]
