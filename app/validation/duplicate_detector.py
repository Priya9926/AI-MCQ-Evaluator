from typing import List, Optional
from app.schemas.generation import RawLLMQuestion
from app.core.config import settings
from app.core.logging import logger


class DuplicateDetector:
    """Detects duplicate or near-duplicate candidate questions based on semantic/text similarity."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold or settings.DUPLICATE_SIMILARITY_THRESHOLD

    def is_duplicate(self, candidate: RawLLMQuestion, accepted_questions: List[RawLLMQuestion]) -> bool:
        cand_tokens = set(candidate.question.lower().split())
        if not cand_tokens:
            return False

        for accepted in accepted_questions:
            acc_tokens = set(accepted.question.lower().split())
            if not acc_tokens:
                continue

            intersection = cand_tokens.intersection(acc_tokens)
            union = cand_tokens.union(acc_tokens)
            jaccard_sim = len(intersection) / len(union) if union else 0.0

            if jaccard_sim >= self.threshold:
                logger.warning(
                    f"Rejected question: High duplicate similarity ({jaccard_sim:.2f} >= {self.threshold:.2f}) "
                    f"with accepted question: '{accepted.question[:50]}...'"
                )
                return True

        return False
