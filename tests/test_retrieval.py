import pytest
from app.schemas.document import DocumentChunk
from app.schemas.metadata import ChunkMetadata
from app.vectorstore.chroma_store import ChromaVectorStore
from app.embeddings.embedding_service import EmbeddingService
from app.rag.retriever import TopicAwareRetriever


def test_retrieval_topic_isolation(tmp_path):
    chroma_dir = str(tmp_path / "chroma")
    vector_store = ChromaVectorStore(collection_name="test_retrieval_collection", persist_directory=chroma_dir)
    embedding_service = EmbeddingService()
    retriever = TopicAwareRetriever(vector_store=vector_store, embedding_service=embedding_service)

    # Ingest Acceleration chunk
    acc_chunk = DocumentChunk(
        chunk_id="acc_001",
        text="Acceleration is the rate of change of velocity. Formula a = dv/dt.",
        metadata=ChunkMetadata(
            chunk_id="acc_001",
            subject="Physics",
            chapter="Mechanics",
            topic="Acceleration",
            subtopic="Definition",
            source_document="physics.pdf",
            page_number=1
        )
    )

    # Ingest Heat chunk
    heat_chunk = DocumentChunk(
        chunk_id="heat_001",
        text="Heat is energy transferred between systems due to temperature difference.",
        metadata=ChunkMetadata(
            chunk_id="heat_001",
            subject="Physics",
            chapter="Thermodynamics",
            topic="Heat",
            subtopic="Definition",
            source_document="physics.pdf",
            page_number=10
        )
    )

    chunks = [acc_chunk, heat_chunk]
    embeddings = embedding_service.embed_texts([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    # Search specifically for Acceleration
    results = retriever.retrieve(
        query_text="What is rate of change of velocity?",
        metadata_filter={"subject": "Physics", "chapter": "Mechanics", "topic": "Acceleration"},
        top_k=5
    )

    assert len(results) == 1
    assert results[0].chunk_id == "acc_001"
    assert results[0].metadata.topic == "Acceleration"
    assert "Heat" not in [r.metadata.topic for r in results]
