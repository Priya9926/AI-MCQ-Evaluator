import pytest
from app.ingestion.topic_chunker import TopicChunker


def test_topic_aware_chunking():
    annotated_blocks = [
        {"text": "Velocity is displacement per unit time.", "subject": "Physics", "chapter": "Mechanics", "topic": "Velocity", "subtopic": "Definition", "page_number": 1},
        {"text": "Velocity formula is v = d/t.", "subject": "Physics", "chapter": "Mechanics", "topic": "Velocity", "subtopic": "Formula", "page_number": 1},
        {"text": "Acceleration is rate of change of velocity.", "subject": "Physics", "chapter": "Mechanics", "topic": "Acceleration", "subtopic": "Definition", "page_number": 2},
        {"text": "Acceleration formula is a = dv/dt.", "subject": "Physics", "chapter": "Mechanics", "topic": "Acceleration", "subtopic": "Formula", "page_number": 2},
    ]

    chunker = TopicChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_annotated_blocks(annotated_blocks, source_document="test_doc.txt")

    assert len(chunks) == 4
    # Ensure Velocity and Acceleration chunks are distinct
    velocity_chunks = [c for c in chunks if c.metadata.topic == "Velocity"]
    acceleration_chunks = [c for c in chunks if c.metadata.topic == "Acceleration"]

    assert len(velocity_chunks) == 2
    assert len(acceleration_chunks) == 2
    for c in velocity_chunks:
        assert "acceleration" not in c.metadata.topic.lower()
