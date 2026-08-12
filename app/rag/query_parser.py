from typing import Tuple, Dict, Any
from app.schemas.question import QuestionGenerationRequest


class QueryParser:
    """Parses incoming question generation requests into query strings and metadata filter rules."""

    @staticmethod
    def parse_request(request: QuestionGenerationRequest) -> Tuple[str, Dict[str, Any]]:
        query_text = f"{request.subject} {request.chapter} {request.topic}"
        if request.subtopic:
            query_text += f" {request.subtopic}"

        metadata_filter: Dict[str, Any] = {}

        if request.subject:
            metadata_filter["subject"] = request.subject
        if request.chapter:
            metadata_filter["chapter"] = request.chapter
        if request.topic:
            metadata_filter["topic"] = request.topic
        if request.subtopic:
            metadata_filter["subtopic"] = request.subtopic
        if request.document_id:
            metadata_filter["source_document"] = request.document_id

        return query_text, metadata_filter
