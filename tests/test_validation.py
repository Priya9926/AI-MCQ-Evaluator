import pytest
from app.schemas.generation import RawLLMQuestion
from app.schemas.question import QuestionGenerationRequest
from app.schemas.document import DocumentChunk
from app.schemas.metadata import ChunkMetadata
from app.validation.answer_validator import AnswerValidator
from app.validation.groundedness import GroundednessValidator
from app.validation.duplicate_detector import DuplicateDetector
from app.validation.pipeline import ValidationPipeline


def test_answer_validator():
    validator = AnswerValidator()

    # Valid candidate
    valid_q = RawLLMQuestion(
        question="What is uniform acceleration?",
        options={"A": "Option 1", "B": "Option 2", "C": "Option 3", "D": "Option 4"},
        correct_answer="A",
        explanation="Derived from text",
        source_chunk_ids=["chunk_1"]
    )
    assert validator.validate(valid_q) is True

    # Invalid option key
    invalid_q = RawLLMQuestion(
        question="What is acceleration?",
        options={"A": "Option 1", "B": "Option 2"},
        correct_answer="A",
        explanation="Explanation",
        source_chunk_ids=["chunk_1"]
    )
    assert validator.validate(invalid_q) is False


def test_duplicate_detector():
    detector = DuplicateDetector(threshold=0.8)

    q1 = RawLLMQuestion(
        question="What is acceleration defined as?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct_answer="A",
        explanation="exp",
        source_chunk_ids=["c1"]
    )

    q2 = RawLLMQuestion(
        question="What is acceleration defined as?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct_answer="A",
        explanation="exp",
        source_chunk_ids=["c1"]
    )

    assert detector.is_duplicate(q2, [q1]) is True
