import re
from typing import List, Dict, Any, Union
from app.schemas.question import QuestionGenerationRequest
from app.schemas.document import DocumentChunk
from app.schemas.generation import RawLLMResponse, RawLLMQuestion
from app.generation.gemini_client import GeminiClient
from app.generation.prompt_builder import PromptBuilder
from app.core.logging import logger


class QuestionGenerator:
    """Invokes Gemini LLM to generate candidate questions grounded in retrieved source context."""

    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def generate_candidates(
        self,
        request: QuestionGenerationRequest,
        context_str: str,
        retrieved_chunks: List[DocumentChunk]
    ) -> List[RawLLMQuestion]:
        if not retrieved_chunks:
            logger.warning("No retrieved chunks provided to QuestionGenerator.")
            return []

        system_instruction = PromptBuilder.get_system_instruction()
        prompt = PromptBuilder.build_generation_prompt(request, context_str)

        json_data = self.gemini_client.generate_json(prompt, system_instruction)

        if json_data:
            # Handle direct list of questions
            if isinstance(json_data, list):
                json_data = {"questions": json_data}
            elif isinstance(json_data, dict):
                if "questions" not in json_data:
                    for key in ["items", "data", "results", "output"]:
                        if key in json_data and isinstance(json_data[key], list):
                            json_data["questions"] = json_data[key]
                            break

            if "questions" in json_data and isinstance(json_data["questions"], list):
                try:
                    # Sanitize and ensure valid structure for each raw question
                    sanitized_list = []
                    for q in json_data["questions"]:
                        if not isinstance(q, dict):
                            continue
                        # Ensure options is a dict of A, B, C, D
                        opts = q.get("options", {})
                        if isinstance(opts, list):
                            keys = ["A", "B", "C", "D"]
                            opts = {keys[i]: str(opt) for i, opt in enumerate(opts[:4]) if i < len(keys)}
                        # Ensure source_chunk_ids has fallback
                        cids = q.get("source_chunk_ids", [])
                        if not cids:
                            cids = [retrieved_chunks[0].chunk_id]

                        q["options"] = opts
                        q["source_chunk_ids"] = cids
                        sanitized_list.append(q)

                    raw_response = RawLLMResponse(questions=[RawLLMQuestion(**q) for q in sanitized_list])
                    logger.info(f"Gemini LLM generated {len(raw_response.questions)} grounded candidate questions.")
                    return raw_response.questions
                except Exception as e:
                    logger.error(f"Failed to parse Gemini JSON output into RawLLMResponse: {e}")

        # Fallback generator for offline/testing mode when Gemini API is unconfigured
        logger.info("Using heuristic candidate generator fallback (offline/testing mode).")
        return self._rule_based_fallback_generator(request, retrieved_chunks)

    def _rule_based_fallback_generator(
        self,
        request: QuestionGenerationRequest,
        chunks: List[DocumentChunk]
    ) -> List[RawLLMQuestion]:
        """Generates grounded questions from retrieved chunk sentences for offline/testing environments."""
        candidates: List[RawLLMQuestion] = []

        for idx, chunk in enumerate(chunks):
            sentences = [s.strip() for s in re.split(r'[.!?\n]', chunk.text) if len(s.strip()) > 20]
            for s_idx, sentence in enumerate(sentences):
                if len(candidates) >= request.count:
                    break

                q_text = f"Regarding {chunk.metadata.topic}, which of the following is accurate according to the material?"
                opt_a = sentence
                opt_b = f"The statement '{sentence}' applies only under reversed conditions."
                opt_c = f"This concept is defined independently without reference to {chunk.metadata.topic}."
                opt_d = f"The source text does not state that {sentence[:40]}..."

                candidates.append(
                    RawLLMQuestion(
                        question=q_text,
                        options={"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d},
                        correct_answer="A",
                        explanation=f"Directly stated in source chunk {chunk.chunk_id}: '{sentence}'.",
                        subtopic=chunk.metadata.subtopic,
                        difficulty=request.difficulty.value.lower(),
                        source_chunk_ids=[chunk.chunk_id]
                    )
                )

        return candidates
