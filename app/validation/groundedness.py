import re
from typing import List, Dict
from app.schemas.generation import RawLLMQuestion
from app.schemas.document import DocumentChunk
from app.core.logging import logger


class GroundednessValidator:
    """Validates that candidate questions and answers are strictly grounded in retrieved source chunks."""

    def validate(self, candidate: RawLLMQuestion, chunk_map: Dict[str, DocumentChunk]) -> bool:
        # Step 1: Check source chunk IDs validity
        if not candidate.source_chunk_ids:
            logger.warning("Rejected question: Missing source_chunk_ids.")
            return False

        valid_chunk_texts = []
        for cid in candidate.source_chunk_ids:
            if cid in chunk_map:
                valid_chunk_texts.append(chunk_map[cid].text.lower())
            else:
                # Check if chunk ID partially matches any retrieved chunk ID
                matching_chunks = [c.text.lower() for c_id, c in chunk_map.items() if c_id.startswith(cid) or cid.startswith(c_id)]
                if matching_chunks:
                    valid_chunk_texts.extend(matching_chunks)

        if not valid_chunk_texts:
            logger.warning(f"Rejected question: Specified chunk IDs {candidate.source_chunk_ids} not found in retrieved set.")
            return False

        combined_source_text = " ".join(valid_chunk_texts)

        # Step 2: Verify correct answer text overlap with source context
        correct_opt_text = candidate.options.get(candidate.correct_answer, "").strip().lower()
        if not correct_opt_text:
            logger.warning("Rejected question: Empty correct option text.")
            return False

        # Extract key content words from correct answer
        keywords = [w for w in re.findall(r'\b\w{4,}\b', correct_opt_text) if w not in {"which", "that", "this", "what", "where", "with", "from", "have"}]

        if keywords:
            supported_keywords = [kw for kw in keywords if kw in combined_source_text]
            support_ratio = len(supported_keywords) / len(keywords)
            if support_ratio < 0.3:
                logger.warning(f"Rejected question: Low groundedness support ratio ({support_ratio:.2f}) for answer '{correct_opt_text}'.")
                return False

        return True
