from typing import List
from app.schemas.document import DocumentChunk


class ContextBuilder:
    """Formats retrieved chunks into structured, delimited context strings for LLM prompt ingestion."""

    @staticmethod
    def build_context_block(chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return "NO RELEVANT SOURCE MATERIAL FOUND FOR THE REQUESTED TOPIC."

        formatted_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            block = (
                f"--- SOURCE CHUNK [{idx}] ---\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Subject: {chunk.metadata.subject}\n"
                f"Chapter: {chunk.metadata.chapter}\n"
                f"Topic: {chunk.metadata.topic}\n"
                f"Subtopic: {chunk.metadata.subtopic}\n"
                f"Page Number: {chunk.metadata.page_number}\n"
                f"Source Document: {chunk.metadata.source_document}\n"
                f"Text:\n{chunk.text.strip()}\n"
            )
            formatted_chunks.append(block)

        return "\n\n".join(formatted_chunks)
