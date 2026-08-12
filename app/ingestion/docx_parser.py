from docx import Document
from typing import List
from app.schemas.document import ExtractedElement
from app.core.logging import logger


class DOCXParser:
    """Parses DOCX files using python-docx, preserving style headings and paragraph order."""

    def parse(self, file_path: str) -> List[ExtractedElement]:
        elements: List[ExtractedElement] = []
        try:
            doc = Document(file_path)
            for idx, p in enumerate(doc.paragraphs, start=1):
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name.lower() if (p.style and p.style.name) else ""
                heading_level = None
                element_type = "paragraph"
                is_bold = any(run.bold for run in p.runs if run.bold)

                if "heading 1" in style_name:
                    heading_level = 1
                    element_type = "heading"
                elif "heading 2" in style_name:
                    heading_level = 2
                    element_type = "heading"
                elif "heading 3" in style_name:
                    heading_level = 3
                    element_type = "heading"
                elif is_bold and len(text) < 100:
                    heading_level = 3
                    element_type = "heading"

                elements.append(
                    ExtractedElement(
                        text=text,
                        page_number=1,  # DOCX files do not have explicit page breaks without layout engine
                        is_bold=is_bold,
                        heading_level=heading_level,
                        element_type=element_type
                    )
                )
            logger.info(f"Successfully extracted {len(elements)} elements from DOCX: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing DOCX file {file_path}: {e}")
            raise e
        return elements
