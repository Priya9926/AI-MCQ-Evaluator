from typing import List, Dict, Any, Optional
from app.schemas.document import DocumentChunk
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.base import VectorStoreBase
from app.core.config import settings
from app.core.logging import logger


class TopicAwareRetriever:
    """Retrieves document chunks by enforcing metadata filter priority before vector similarity search."""

    def __init__(self, vector_store: VectorStoreBase, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query_text: str,
        metadata_filter: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        k = top_k or settings.TOP_K

        # Step 1: Generate Query Embedding
        query_embedding = self.embedding_service.embed_query(query_text)

        # Step 2: Perform Metadata Filtered Vector Search
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            metadata_filter=metadata_filter,
            top_k=k
        )

        # Step 3: Strict Topic Isolation Check
        validated_chunks: List[DocumentChunk] = []
        req_subject = metadata_filter.get("subject", "").lower()
        req_chapter = metadata_filter.get("chapter", "").lower()
        req_topic = metadata_filter.get("topic", "").lower()

        for chunk in retrieved_chunks:
            meta = chunk.metadata
            # Enforce topic boundary consistency
            if req_subject and meta.subject.lower() != req_subject:
                logger.warning(f"Rejected chunk {chunk.chunk_id}: Subject mismatch '{meta.subject}' != '{req_subject}'")
                continue
            if req_chapter and meta.chapter.lower() != req_chapter:
                logger.warning(f"Rejected chunk {chunk.chunk_id}: Chapter mismatch '{meta.chapter}' != '{req_chapter}'")
                continue
            if req_topic and meta.topic.lower() != req_topic:
                logger.warning(f"Rejected chunk {chunk.chunk_id}: Topic mismatch '{meta.topic}' != '{req_topic}'")
                continue

            validated_chunks.append(chunk)

        logger.info(f"Retriever returned {len(validated_chunks)} strictly validated topic chunks for topic '{req_topic}'")
        return validated_chunks
