import json
import re
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging import logger

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-3.5-flash"
]


class GeminiClient:
    """Wrapper around modern google-genai SDK with automatic model fallback."""

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.client = None
        if self.api_key and self.api_key != "mock_key_for_testing":
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized GeminiClient with primary model '{self.model_name}'")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini SDK client: {e}")

    def generate_json(self, prompt: str, system_instruction: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            logger.warning("Gemini Client not available or using mock key.")
            return None

        # Build candidate model list with primary model first
        models_to_try: List[str] = [self.model_name]
        for fm in FALLBACK_MODELS:
            if fm not in models_to_try:
                models_to_try.append(fm)

        last_error = None
        for model in models_to_try:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    response_mime_type="application/json"
                )
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )

                text = response.text
                if not text:
                    continue

                # Clean potential markdown block markers ```json ... ```
                cleaned_text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
                cleaned_text = re.sub(r'```\s*$', '', cleaned_text.strip(), flags=re.MULTILINE)

                result = json.loads(cleaned_text)
                if result:
                    if model != self.model_name:
                        logger.info(f"Gemini generation succeeded with fallback model '{model}'")
                    return result
            except Exception as e:
                last_error = e
                logger.warning(f"Model '{model}' failed: {e}. Trying fallback model if available...")

        logger.error(f"All Gemini models failed. Last error: {last_error}")
        return None
