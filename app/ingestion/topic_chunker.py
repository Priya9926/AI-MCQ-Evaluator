from typing import List, Dict, Any, Optional
from app.schemas.document import DocumentChunk
from app.schemas.metadata import ChunkMetadata
from app.utils.helpers import generate_chunk_id
from app.core.config import settings
from app.core.logging import logger


class TopicChunker:
    """
    Performs topic-aware semantic chunking enforcing strict topic boundary isolation.
    chunk_size is measured in WORDS (not characters).
    Real educational PDFs benefit from smaller chunk_size (150-300 words) for granular retrieval.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_annotated_blocks(
        self, annotated_blocks: List[Dict[str, Any]], source_document: str
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []

        if not annotated_blocks:
            return chunks

        # Group contiguous blocks by (subject, chapter, topic, subtopic)
        grouped_sections: List[Dict[str, Any]] = []
        current_group: Optional[Dict[str, Any]] = None

        for block in annotated_blocks:
            key = (block["subject"], block["chapter"], block["topic"], block["subtopic"])
            if current_group is None or (
                current_group["subject"],
                current_group["chapter"],
                current_group["topic"],
                current_group["subtopic"]
            ) != key:
                if current_group:
                    grouped_sections.append(current_group)
                current_group = {
                    "subject": block["subject"],
                    "chapter": block["chapter"],
                    "topic": block["topic"],
                    "subtopic": block["subtopic"],
                    "blocks": [block]
                }
            else:
                current_group["blocks"].append(block)

        if current_group:
            grouped_sections.append(current_group)

        chunk_counter = 1

        # Process each group independently to guarantee zero cross-topic boundary pollution
        for group in grouped_sections:
            subject = group["subject"]
            chapter = group["chapter"]
            topic = group["topic"]
            subtopic = group["subtopic"]
            blocks = group["blocks"]

            full_text = "\n".join([b["text"] for b in blocks])
            page_numbers = [b["page_number"] for b in blocks]
            primary_page = page_numbers[0] if page_numbers else 1

            # If full_text fits within chunk_size, create a single chunk
            if len(full_text) <= self.chunk_size:
                cid = generate_chunk_id(subject, chapter, topic, chunk_counter, doc_id=source_document)
                meta = ChunkMetadata(
                    chunk_id=cid,
                    subject=subject,
                    chapter=chapter,
                    topic=topic,
                    subtopic=subtopic,
                    source_document=source_document,
                    page_number=primary_page
                )
                chunks.append(DocumentChunk(chunk_id=cid, text=full_text, metadata=meta))
                chunk_counter += 1
            else:
                # Sub-chunk within group boundary using sentence/paragraph boundaries
                words = full_text.split()
                start_idx = 0
                step = max(1, self.chunk_size - self.chunk_overlap)

                while start_idx < len(words):
                    end_idx = min(start_idx + self.chunk_size, len(words))
                    chunk_text = " ".join(words[start_idx:end_idx])

                    if chunk_text.strip():
                        cid = generate_chunk_id(subject, chapter, topic, chunk_counter, doc_id=source_document)
                        meta = ChunkMetadata(
                            chunk_id=cid,
                            subject=subject,
                            chapter=chapter,
                            topic=topic,
                            subtopic=subtopic,
                            source_document=source_document,
                            page_number=primary_page
                        )
                        chunks.append(DocumentChunk(chunk_id=cid, text=chunk_text, metadata=meta))
                        chunk_counter += 1

                    if end_idx >= len(words):
                        break
                    start_idx += step

        logger.info(f"Generated {len(chunks)} topic-aware semantic chunks for document '{source_document}'")
        return chunks
