from app.schemas.generation import RawLLMQuestion
from app.core.logging import logger


class AnswerValidator:
    """Validates answer options structure, uniqueness of choices, and presence of single correct key."""

    def validate(self, candidate: RawLLMQuestion) -> bool:
        # 1. Correct answer key must be one of 'A', 'B', 'C', 'D'
        if candidate.correct_answer not in {'A', 'B', 'C', 'D'}:
            logger.warning(f"Rejected question: Invalid correct answer key '{candidate.correct_answer}'.")
            return False

        # 2. Options dictionary must contain A, B, C, D
        keys = set(candidate.options.keys())
        if not {'A', 'B', 'C', 'D'}.issubset(keys):
            logger.warning(f"Rejected question: Incomplete option keys {keys}.")
            return False

        # 3. All options must be non-empty strings
        option_texts = [candidate.options[k].strip().lower() for k in ['A', 'B', 'C', 'D']]
        if any(not opt for opt in option_texts):
            logger.warning("Rejected question: One or more options are empty.")
            return False

        # 4. No two options can be identical
        if len(set(option_texts)) < len(option_texts):
            logger.warning("Rejected question: Duplicate option choices found.")
            return False

        return True
