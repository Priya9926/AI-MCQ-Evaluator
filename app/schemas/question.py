from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MCQQuestion(BaseModel):
    question_id: str = Field(..., description="Unique ID for the generated question")
    question: str = Field(..., description="The question text")
    options: Dict[str, str] = Field(..., description="Keyed options dict, e.g. {'A': '...', 'B': '...'}")
    correct_answer: str = Field(..., description="Correct option key, e.g., 'A'")
    explanation: str = Field(..., description="Detailed explanation grounded in the text")
    subject: str = Field(..., description="Subject name")
    chapter: str = Field(..., description="Chapter name")
    topic: str = Field(..., description="Topic name")
    subtopic: str = Field(default="General", description="Subtopic name")
    difficulty: str = Field(default="medium", description="Question difficulty: easy, medium, hard")
    source_chunk_ids: List[str] = Field(..., description="Mandatory IDs of retrieved source chunks supporting this question")


class QuestionGenerationRequest(BaseModel):
    document_id: Optional[str] = Field(default=None, description="Optional document ID to restrict query scope")
    subject: str = Field(..., description="Target Subject")
    chapter: str = Field(..., description="Target Chapter")
    topic: str = Field(..., description="Target Topic")
    subtopic: Optional[str] = Field(default=None, description="Optional target Subtopic")
    question_type: QuestionType = Field(default=QuestionType.MCQ, description="Question type (default: mcq)")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM, description="Difficulty level")
    count: int = Field(default=5, ge=1, le=50, description="Number of questions requested")


class QuestionGenerationResponse(BaseModel):
    questions: List[MCQQuestion]
    total_generated: int
    total_requested: int
    rejected_count: int
    subject: str
    chapter: str
    topic: str
    subtopic: Optional[str] = None
