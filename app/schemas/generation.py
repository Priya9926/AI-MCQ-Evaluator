from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class RawLLMQuestion(BaseModel):
    question: str = Field(..., description="Question stem")
    options: Dict[str, str] = Field(..., description="Dictionary of options e.g. {'A': 'val', 'B': 'val'}")
    correct_answer: str = Field(..., description="Key corresponding to correct option")
    explanation: str = Field(..., description="Explanation of correct answer grounded in source context")
    subtopic: Optional[str] = Field(default="General", description="Subtopic identifier")
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs of source chunks used")


class RawLLMResponse(BaseModel):
    questions: List[RawLLMQuestion]
