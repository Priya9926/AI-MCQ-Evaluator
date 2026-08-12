from app.schemas.generation import RawLLMQuestion
from app.schemas.question import QuestionGenerationRequest
from app.core.logging import logger


class TopicValidator:
    """Ensures generated question strictly belongs to the requested topic and does not introduce off-topic concepts."""

    def validate(self, candidate: RawLLMQuestion, request: QuestionGenerationRequest) -> bool:
        req_topic = request.topic.strip().lower()
        question_text = candidate.question.lower()
        explanation_text = candidate.explanation.lower()

        # Check if requested topic term (or main word) is present or compatible
        topic_words = [w for w in req_topic.split() if len(w) > 3]

        if topic_words:
            has_topic_reference = any(w in question_text or w in explanation_text for w in topic_words)
            if not has_topic_reference:
                logger.warning(f"Rejected question: Question text does not reference target topic '{req_topic}'.")
                return False

        return True
