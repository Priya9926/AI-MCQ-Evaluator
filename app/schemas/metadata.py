from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_id: str = Field(..., description="Unique ID for the chunk")
    subject: str = Field(..., description="Academic subject name (e.g., Physics)")
    class_level: str = Field(default="General", description="Target grade/class level")
    chapter: str = Field(..., description="Chapter name")
    topic: str = Field(..., description="Topic name within chapter")
    subtopic: str = Field(default="General", description="Subtopic name within topic")
    source_document: str = Field(..., description="Filename of original document")
    page_number: int = Field(default=1, description="Page number where text originated")
    language: str = Field(default="English", description="Language of text")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class SubtopicNode(BaseModel):
    name: str
    chunk_ids: List[str] = Field(default_factory=list)


class TopicNode(BaseModel):
    name: str
    subtopics: List[str] = Field(default_factory=list)


class ChapterNode(BaseModel):
    name: str
    topics: List[TopicNode] = Field(default_factory=list)


class TopicHierarchy(BaseModel):
    subject: str
    chapters: List[ChapterNode] = Field(default_factory=list)
