from typing import List, Tuple
from app.schemas.question import QuestionGenerationRequest
from app.schemas.document import DocumentChunk
from app.rag.query_parser import QueryParser
from app.rag.retriever import TopicAwareRetriever
from app.rag.context_builder import ContextBuilder
from app.core.logging import logger


class RAGPipeline:
    """Orchestrates query parsing, topic-filtered retrieval, and context block construction."""

    def __init__(self, retriever: TopicAwareRetriever):
        self.retriever = retriever

    def retrieve_context(self, request: QuestionGenerationRequest) -> Tuple[str, List[DocumentChunk]]:
        query_text, metadata_filter = QueryParser.parse_request(request)
        chunks = self.retriever.retrieve(
            query_text=query_text,
            metadata_filter=metadata_filter,
            top_k=request.count * 2  # Retrieve extra chunks to allow sufficient material for validation
        )

        context_str = ContextBuilder.build_context_block(chunks)
        logger.info(f"RAGPipeline prepared context block ({len(context_str)} chars) from {len(chunks)} chunks.")
        return context_str, chunks
