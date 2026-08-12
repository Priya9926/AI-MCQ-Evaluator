import pytest
from app.schemas.question import QuestionGenerationRequest
from app.schemas.document import DocumentChunk
from app.schemas.metadata import ChunkMetadata
from app.generation.gemini_client import GeminiClient
from app.generation.question_generator import QuestionGenerator


def test_question_generator_fallback():
    gemini_client = GeminiClient()  # Uses mock fallback
    gemini_client.client = None  # Force fallback behavior even if API key is set in .env
    generator = QuestionGenerator(gemini_client=gemini_client)

    chunk = DocumentChunk(
        chunk_id="physics_mechanics_acceleration_001",
        text="Uniform acceleration occurs when velocity changes at a constant rate over equal time intervals.",
        metadata=ChunkMetadata(
            chunk_id="physics_mechanics_acceleration_001",
            subject="Physics",
            chapter="Mechanics",
            topic="Acceleration",
            subtopic="Uniform Acceleration",
            source_document="doc1.pdf",
            page_number=42
        )
    )

    request = QuestionGenerationRequest(
        subject="Physics",
        chapter="Mechanics",
        topic="Acceleration",
        count=2
    )

    candidates = generator.generate_candidates(request, context_str=chunk.text, retrieved_chunks=[chunk])
    assert len(candidates) >= 1
    assert candidates[0].options is not None
    assert "physics_mechanics_acceleration_001" in candidates[0].source_chunk_ids
