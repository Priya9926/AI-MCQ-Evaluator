from app.schemas.question import QuestionGenerationRequest


class PromptBuilder:
    """Builds strict system prompts and user prompt instructions for Gemini question generation."""

    @staticmethod
    def get_system_instruction() -> str:
        return (
            "You are an educational assessment question generator.\n"
            "Generate questions ONLY from the supplied source context.\n"
            "Do not use outside knowledge.\n"
            "Do not infer facts that are not supported by the context.\n"
            "Every question must belong strictly to the requested subject, chapter, topic, and subtopic.\n"
            "If the provided context does not contain enough information to generate a valid question, do not invent information.\n"
            "Every answer must be directly supported by the source context.\n"
            "You MUST specify the exact source chunk IDs used to derive each question in the 'source_chunk_ids' array.\n"
            "Return structured JSON only matching the schema requested."
        )

    @staticmethod
    def build_generation_prompt(request: QuestionGenerationRequest, context_str: str) -> str:
        subtopic_str = f" → Subtopic: {request.subtopic}" if request.subtopic else ""
        return (
            f"TARGET TOPIC SPECIFICATION:\n"
            f"Subject: {request.subject}\n"
            f"Chapter: {request.chapter}\n"
            f"Topic: {request.topic}{subtopic_str}\n"
            f"Requested Question Type: {request.question_type.value.upper()}\n"
            f"Requested Difficulty: {request.difficulty.value.lower()}\n"
            f"Requested Question Count: {request.count}\n\n"
            f"RETRIEVED SOURCE CONTEXT (USE ONLY THIS CONTENT):\n"
            f"{context_str}\n\n"
            f"INSTRUCTIONS:\n"
            f"Generate exactly {request.count} high-quality Multiple Choice Questions (MCQs).\n"
            f"Each question must have:\n"
            f"1. 'question': Clear question text.\n"
            f"2. 'options': Dictionary with exactly 4 options 'A', 'B', 'C', 'D'.\n"
            f"3. 'correct_answer': Single letter ('A', 'B', 'C', or 'D').\n"
            f"4. 'explanation': Clear explanation citing the source material.\n"
            f"5. 'subtopic': The subtopic name.\n"
            f"6. 'difficulty': '{request.difficulty.value.lower()}'.\n"
            f"7. 'source_chunk_ids': List of chunk IDs (e.g. ['physics_mechanics_acceleration_001']) used for this question.\n\n"
            f"Return JSON output adhering strictly to this schema:\n"
            f"{{\n"
            f'  "questions": [\n'
            f'    {{\n'
            f'      "question": "What is acceleration?",\n'
            f'      "options": {{\n'
            f'        "A": "Option text 1",\n'
            f'        "B": "Option text 2",\n'
            f'        "C": "Option text 3",\n'
            f'        "D": "Option text 4"\n'
            f'      }},\n'
            f'      "correct_answer": "B",\n'
            f'      "explanation": "Derived from source context...",\n'
            f'      "subtopic": "{request.subtopic or "Definition"}",\n'
            f'      "difficulty": "{request.difficulty.value.lower()}",\n'
            f'      "source_chunk_ids": ["physics_mechanics_acceleration_001"]\n'
            f'    }}\n'
            f'  ]\n'
            f"}}\n"
        )
