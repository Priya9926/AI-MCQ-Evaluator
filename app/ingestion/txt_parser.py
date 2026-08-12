import re
from typing import List
from app.schemas.document import ExtractedElement
from app.core.logging import logger


class TXTParser:
    """Parses plain text files, detecting headers via markdown indicators and formatting patterns."""

    def parse(self, file_path: str) -> List[ExtractedElement]:
        elements: List[ExtractedElement] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                text = line.strip()
                if not text:
                    continue

                heading_level = None
                element_type = "paragraph"
                is_bold = False

                # Check markdown headings
                if text.startswith("# "):
                    heading_level = 1
                    element_type = "heading"
                    text = text[2:].strip()
                elif text.startswith("## "):
                    heading_level = 2
                    element_type = "heading"
                    text = text[3:].strip()
                elif text.startswith("### "):
                    heading_level = 3
                    element_type = "heading"
                    text = text[4:].strip()
                # Check standard heading patterns (e.g. "Subject: ...", "Chapter 1: ...", "Topic: ...")
                elif re.match(r'^(Subject|Chapter|Topic|Subtopic)\s*\d*:', text, re.IGNORECASE):
                    element_type = "heading"
                    if text.lower().startswith("subject"):
                        heading_level = 1
                    elif text.lower().startswith("chapter"):
                        heading_level = 2
                    elif text.lower().startswith("topic"):
                        heading_level = 3
                    else:
                        heading_level = 4

                elements.append(
                    ExtractedElement(
                        text=text,
                        page_number=1,
                        is_bold=is_bold,
                        heading_level=heading_level,
                        element_type=element_type
                    )
                )
            logger.info(f"Successfully extracted {len(elements)} elements from TXT: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {e}")
            raise e
        return elements
