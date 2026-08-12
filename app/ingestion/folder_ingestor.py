import os
from typing import List, Dict, Any, Optional
from app.schemas.document import DocumentUploadResponse
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.docx_parser import DOCXParser
from app.ingestion.txt_parser import TXTParser
from app.ingestion.text_cleaner import TextCleaner
from app.ingestion.structure_detector import StructureDetector, _clean_filename_as_chapter
from app.ingestion.topic_chunker import TopicChunker
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaVectorStore
from app.utils.helpers import generate_id
from app.core.config import settings
from app.core.logging import logger


class FolderIngestor:
    """Scans and processes all educational documents inside the upload folder."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.txt_parser = TXTParser()
        self.text_cleaner = TextCleaner()
        self.topic_chunker = TopicChunker()

        # Lazy-import GeminiClient to avoid circular deps; pass in for LLM-assisted detection
        try:
            from app.generation.gemini_client import GeminiClient
            self._gemini_client = GeminiClient()
        except Exception:
            self._gemini_client = None

    def ingest_file(self, file_path: str) -> DocumentUploadResponse:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ['.pdf', '.docx', '.txt']:
            raise ValueError(f"Unsupported file format '{ext}' for file {filename}")

        logger.info(f"Ingesting file: {file_path}")

        # 1. Parse
        if ext == '.pdf':
            elements = self.pdf_parser.parse(file_path)
        elif ext == '.docx':
            elements = self.docx_parser.parse(file_path)
        else:
            elements = self.txt_parser.parse(file_path)

        if not elements:
            raise ValueError(f"File '{filename}' contains no extractable text.")

        # 2. Clean
        cleaned_elements = self.text_cleaner.clean_elements(elements)
        if not cleaned_elements:
            raise ValueError(f"File '{filename}' has no text remaining after cleaning.")

        # 3. Detect Structure & Canonical Academic Subject
        detector = StructureDetector(
            filename=filename,
            gemini_client=self._gemini_client
        )
        hierarchy, annotated_blocks = detector.detect_structure(cleaned_elements)

        # 4. Topic-Aware Semantic Chunking
        doc_id = generate_id("doc")
        chunks = self.topic_chunker.chunk_annotated_blocks(annotated_blocks, source_document=doc_id)

        # 5. Embedding Generation
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            # 6. Vector Store Storage
            self.vector_store.add_chunks(chunks, embeddings)

        max_page = max((e.page_number for e in elements), default=1)

        logger.info(
            f"Successfully indexed document '{filename}' (ID: {doc_id}) "
            f"with {len(chunks)} chunks across {len(hierarchy.chapters)} chapters."
        )
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            total_pages=max_page,
            total_chunks=len(chunks),
            detected_hierarchy=hierarchy,
            message=f"File '{filename}' successfully ingested into RAG store."
        )

    def ingest_all_from_folder(self, folder_path: Optional[str] = None) -> List[DocumentUploadResponse]:
        target_dir = folder_path or settings.UPLOADS_DIR
        os.makedirs(target_dir, exist_ok=True)

        supported_exts = {'.pdf', '.docx', '.txt'}
        results: List[DocumentUploadResponse] = []

        files = [
            f for f in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, f))
            and os.path.splitext(f)[1].lower() in supported_exts
        ]

        logger.info(f"Found {len(files)} eligible documents in '{target_dir}'")

        for f in files:
            file_path = os.path.join(target_dir, f)
            try:
                res = self.ingest_file(file_path)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed to ingest file '{f}': {e}")

        return results
