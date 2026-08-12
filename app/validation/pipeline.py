from typing import List, Dict, Tuple
from app.schemas.generation import RawLLMQuestion
from app.schemas.question import MCQQuestion, QuestionGenerationRequest
from app.schemas.document import DocumentChunk
from app.validation.groundedness import GroundednessValidator
from app.validation.topic_validator import TopicValidator
from app.validation.answer_validator import AnswerValidator
from app.validation.duplicate_detector import DuplicateDetector
from app.utils.helpers import generate_id
from app.core.logging import logger


class ValidationPipeline:
    """Runs generated question candidates through sequential quality, grounding, topic, and duplicate checks."""

    def __init__(self):
        self.groundedness_validator = GroundednessValidator()
        self.topic_validator = TopicValidator()
        self.answer_validator = AnswerValidator()
        self.duplicate_detector = DuplicateDetector()

    def process(
        self,
        candidates: List[RawLLMQuestion],
        request: QuestionGenerationRequest,
        retrieved_chunks: List[DocumentChunk]
    ) -> Tuple[List[MCQQuestion], int]:
        chunk_map: Dict[str, DocumentChunk] = {c.chunk_id: c for c in retrieved_chunks}

        accepted_raw: List[RawLLMQuestion] = []
        final_questions: List[MCQQuestion] = []
        rejected_count = 0

        for candidate in candidates:
            if len(final_questions) >= request.count:
                break

            # 1. Answer Options & Key Structural Validation
            if not self.answer_validator.validate(candidate):
                rejected_count += 1
                continue

            # 2. Topic Match Validation
            if not self.topic_validator.validate(candidate, request):
                rejected_count += 1
                continue

            # 3. Source Groundedness Validation
            if not self.groundedness_validator.validate(candidate, chunk_map):
                rejected_count += 1
                continue

            # 4. Duplicate Check against previously accepted batch
            if self.duplicate_detector.is_duplicate(candidate, accepted_raw):
                rejected_count += 1
                continue

            # Candidate passed all checks!
            accepted_raw.append(candidate)

            q_id = generate_id("q")
            mcq = MCQQuestion(
                question_id=q_id,
                question=candidate.question,
                options=candidate.options,
                correct_answer=candidate.correct_answer,
                explanation=candidate.explanation,
                subject=request.subject,
                chapter=request.chapter,
                topic=request.topic,
                subtopic=candidate.subtopic or request.subtopic or "General",
                difficulty=candidate.difficulty or request.difficulty.value.lower(),
                source_chunk_ids=candidate.source_chunk_ids
            )
            final_questions.append(mcq)

        logger.info(f"Validation Pipeline completed: {len(final_questions)} accepted, {rejected_count} rejected.")
        return final_questions, rejected_count
