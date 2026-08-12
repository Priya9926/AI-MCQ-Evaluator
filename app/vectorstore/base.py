from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.schemas.document import DocumentChunk
from app.schemas.metadata import TopicHierarchy


class VectorStoreBase(ABC):
    """Abstract Vector Store Interface enabling modular replacement (Chroma, Qdrant, Pinecone)."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Stores document chunks with embeddings and searchable metadata."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """Searches vector database after enforcing metadata filters."""
        pass

    @abstractmethod
    def get_topics_hierarchy(self, source_document: Optional[str] = None) -> TopicHierarchy:
        """Retrieves detected Subject -> Chapter -> Topic -> Subtopic tree."""
        pass

    @abstractmethod
    def delete_document(self, source_document: str) -> None:
        """Deletes all chunks associated with a source document."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all vectors and metadata stored."""
        pass
