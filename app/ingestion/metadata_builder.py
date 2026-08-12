from typing import Dict, Any
from app.schemas.metadata import ChunkMetadata


class MetadataBuilder:
    """Builds and validates metadata dictionaries attached to document chunks."""

    @staticmethod
    def build_metadata(
        chunk_id: str,
        subject: str,
        chapter: str,
        topic: str,
        subtopic: str = "General",
        source_document: str = "document.pdf",
        page_number: int = 1,
        class_level: str = "General",
        language: str = "English"
    ) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_id=chunk_id,
            subject=subject.strip(),
            chapter=chapter.strip(),
            topic=topic.strip(),
            subtopic=subtopic.strip(),
            source_document=source_document.strip(),
            page_number=page_number,
            class_level=class_level,
            language=language
        )
