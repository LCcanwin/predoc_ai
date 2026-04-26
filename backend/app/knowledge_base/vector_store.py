"""FAISS vector store for knowledge base retrieval."""

import os
import pickle
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .chunker import chunk_by_semantic_splits, Chunk


class TCMEmbeddings(Embeddings):
    """Embedding wrapper - uses OpenAI or fallback."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazy init of embedding client."""
        if self._client is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._client = OpenAIEmbeddings(model=self.model_name)
            except Exception:
                # Fallback: use deterministic embeddings for testing
                self._client = None
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        if client:
            return client.embed_documents(texts)
        # Fallback: deterministic hash-based embeddings
        return [self._deterministic_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        if client:
            return client.embed_query(text)
        return self._deterministic_embedding(text)

    def _deterministic_embedding(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embeddings from text hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # Normalize to unit vector
        import math
        magnitude = math.sqrt(sum(b * b for b in h))
        return [b / magnitude if magnitude > 0 else 0 for b in h]


class VectorStore:
    """FAISS vector store wrapper."""

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._store: Optional[FAISS] = None
        self._chunks: list[Chunk] = []

    def build_from_chunks(self, chunks: list[Chunk], embeddings: Optional[Embeddings] = None) -> None:
        """Build vector store from chunks."""
        if embeddings is None:
            embeddings = TCMEmbeddings()

        docs = [
            Document(page_content=chunk.content, metadata={"source": chunk.source, "chunk_index": chunk.chunk_index})
            for chunk in chunks
        ]

        self._store = FAISS.from_documents(docs, embeddings)
        self._chunks = chunks

        if self.persist_path:
            self.save()

    def save(self) -> None:
        """Persist vector store to disk."""
        if self._store and self.persist_path:
            Path(self.persist_path).mkdir(parents=True, exist_ok=True)
            self._store.save_local(self.persist_path)
            chunks_path = Path(self.persist_path) / "chunks.pkl"
            with open(chunks_path, "wb") as f:
                pickle.dump(self._chunks, f)

    def load(self) -> bool:
        """Load vector store from disk."""
        if not self.persist_path:
            return False

        store_path = Path(self.persist_path)
        chunks_path = store_path / "chunks.pkl"
        if not store_path.exists():
            return False

        try:
            self._store = FAISS.load_local(
                self.persist_path,
                TCMEmbeddings(),
                allow_dangerous_deserialization=True
            )
            if chunks_path.exists():
                with open(chunks_path, "rb") as f:
                    self._chunks = pickle.load(f)
            else:
                self._chunks = []
            return True
        except Exception:
            return False

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search for similar documents."""
        if not self._store:
            return []
        return self._store.similarity_search(query, k=k)

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks


def build_vector_store_from_directory(
    directory: str,
    persist_path: str,
    chunk_size: int = 100,
) -> VectorStore:
    """
    Build and persist a vector store from documents in a directory.

    Args:
        directory: Path to knowledge base files
        persist_path: Path to store FAISS index
        chunk_size: Minimum chunk size

    Returns:
        VectorStore instance
    """
    from .loader import load_documents

    chunks = []
    for filename, text in load_documents(directory):
        for chunk in chunk_by_semantic_splits(text, filename, min_chunk_size=chunk_size):
            chunks.append(chunk)

    vector_store = VectorStore(persist_path=persist_path)
    vector_store.build_from_chunks(chunks)

    return vector_store
