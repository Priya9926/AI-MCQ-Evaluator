import re
import uuid
import hashlib
from typing import List, Dict, Any, Optional


def generate_id(prefix: str = "doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text.strip('_')


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def generate_chunk_id(subject: str, chapter: str, topic: str, index: int, doc_id: Optional[str] = None) -> str:
    s_slug = slugify(subject)
    c_slug = slugify(chapter)
    t_slug = slugify(topic)
    if doc_id:
        return f"{doc_id}_{s_slug}_{c_slug}_{t_slug}_{index:03d}"
    return f"{s_slug}_{c_slug}_{t_slug}_{index:03d}"
