"""
Offline indexing, shared by every RAG type:
    load files -> chunk -> embed -> store
"""

from rag_titan.loaders.loader_registry import load_document
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class IndexingPipeline:
    """
    Prepares files so they can be searched.

    Args:
        chunker: a BaseChunker.
        embedder: a BaseEmbedder.
        store: a BaseStore.
    """

    def __init__(self, chunker, embedder, store):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def run(self, file_paths, reset=True):
        """
        Index the files.

        Args:
            file_paths: list of file paths to index.
            reset: when True, clear the store before adding new chunks.

        Returns:
            The number of chunks stored.
        """
        documents = self.load_documents(file_paths)

        chunks = self.chunker.chunk(documents)
        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            logger.warning("No chunks created, nothing to index")
            return 0

        vectors = self.embedder.embed_many([chunk["text"] for chunk in chunks])

        if reset:
            self.store.clear()

        self.store.add(chunks, vectors)
        logger.info("Stored %d chunks in the vector store", self.store.count())

        return len(chunks)

    def load_documents(self, file_paths):
        """Load every file and skip the ones without any text."""
        documents = []

        for path in file_paths:
            document = load_document(path)

            if not document["text"].strip():
                logger.warning("Skipping empty document: %s", document["source"])
                continue

            word_count = len(document["text"].split())
            logger.info("Loaded %s (%d words)", document["source"], word_count)
            documents.append(document)

        return documents
