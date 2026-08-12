from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.metadata import ChunkMetadata, TopicHierarchy


class ExtractedElement(BaseModel):
    text: str
    page_number: int = 1
    font_size: Optional[float] = None
    is_bold: bool = False
    heading_level: Optional[int] = None  # 1 for H1, 2 for H2, etc.
    element_type: str = "paragraph"  # "heading", "paragraph", "list_item"


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    detected_hierarchy: TopicHierarchy
    message: str = "Document successfully ingested and indexed"


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    subject: str
    chapters_count: int
    chunks_count: int
    hierarchy: TopicHierarchy
