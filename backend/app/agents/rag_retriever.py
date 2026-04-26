"""RAG retriever for knowledge base queries."""

from typing import Optional
from langchain_core.documents import Document

from ..knowledge_base.vector_store import VectorStore


class RAGRetriever:
    """Retrieves relevant knowledge base content for queries."""

    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query string

        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=self.top_k)

    def format_retrieved_docs(self, docs: list[Document]) -> str:
        """Format retrieved documents for prompt injection."""
        if not docs:
            return "未找到相关知识库内容。"

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            formatted.append(f"[文档{i}] (来源: {source})\n{content}")

        return "\n\n".join(formatted)
