import numpy as np
from typing import List
from google import genai
from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    """Service to generate embeddings using google-genai SDK text-embedding-004 with deterministic fallback."""

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.EMBEDDING_MODEL
        self._client = None
        if self.api_key and self.api_key != "mock_key_for_testing":
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for embeddings: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._client:
            try:
                # Use modern google-genai SDK
                embeddings = []
                for text in texts:
                    res = self._client.models.embed_content(
                        model=self.model_name,
                        contents=text
                    )
                    # embed_content returns response with embedding
                    if hasattr(res, 'embedding') and hasattr(res.embedding, 'values'):
                        embeddings.append(list(res.embedding.values))
                    elif isinstance(res, dict) and 'embedding' in res:
                        embeddings.append(res['embedding']['values'])
                    else:
                        embeddings.append(self._fallback_embed(text))
                return embeddings
            except Exception as e:
                logger.warning(f"Gemini embedding API call failed: {e}. Using fallback embeddings.")

        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def _fallback_embed(self, text: str, dim: int = 768) -> List[float]:
        """Deterministic TF-IDF hash vectorizer for testing or offline operation."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            idx = sum(ord(c) for c in word) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()
