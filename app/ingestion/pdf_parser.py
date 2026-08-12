import os
import re
from typing import List
from collections import Counter
import pymupdf as fitz
from app.schemas.document import ExtractedElement
from app.core.logging import logger


class PDFParser:
    """
    Parses PDF documents using PyMuPDF.
    Uses adaptive font-size thresholds and preserves native span text without artificial splitting.
    """

    def parse(self, file_path: str) -> List[ExtractedElement]:
        elements: List[ExtractedElement] = []
        try:
            doc = fitz.open(file_path)

            # --- Pass 1: collect font sizes to compute adaptive thresholds ---
            all_font_sizes: List[float] = []
            for page in doc:
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") == 0:
                        for line in b.get("lines", []):
                            if isinstance(line, dict):
                                for span in line.get("spans", []):
                                    if isinstance(span, dict):
                                        fs = float(span.get("size", 0.0))
                                        if fs > 0:
                                            all_font_sizes.append(round(fs, 1))

            # Determine body font size = most common size
            if all_font_sizes:
                size_counter = Counter(all_font_sizes)
                body_size = size_counter.most_common(1)[0][0]
            else:
                body_size = 11.0

            # H1 threshold: body + 4pt minimum, H2: body + 1.5pt minimum
            h1_threshold = body_size + 4.0
            h2_threshold = body_size + 1.5
            logger.info(
                f"PDF font thresholds for '{os.path.basename(file_path)}': "
                f"body={body_size:.1f}pt, H1>={h1_threshold:.1f}pt, H2>={h2_threshold:.1f}pt"
            )

            # --- Pass 2: Extract elements with adaptive heading classification ---
            for page_num, page in enumerate(doc, start=1):
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
                for b in blocks:
                    if not isinstance(b, dict) or b.get("type") != 0:
                        continue
                    for line in b.get("lines", []):
                        if not isinstance(line, dict):
                            continue

                        # Preserve span text directly without artificial space injection between character spans
                        raw_spans_text = []
                        max_font_size = 0.0
                        is_bold = False

                        for span in line.get("spans", []):
                            if not isinstance(span, dict):
                                continue
                            span_text = str(span.get("text", ""))
                            font_size = float(span.get("size", 0.0))
                            flags = int(span.get("flags", 0))
                            font_name = str(span.get("font", "")).lower()

                            if span_text:
                                raw_spans_text.append(span_text)
                                if font_size > max_font_size:
                                    max_font_size = font_size
                                if (flags & 16) or "bold" in font_name or "black" in font_name:
                                    is_bold = True

                        full_line = "".join(raw_spans_text)
                        # Normalize multiple whitespace characters
                        line_text = re.sub(r'[ \t]+', ' ', full_line).strip()
                        if not line_text or len(line_text) < 2:
                            continue

                        # Adaptive heading classification
                        heading_level = None
                        element_type = "paragraph"

                        if max_font_size >= h1_threshold and len(line_text) < 150:
                            heading_level = 1
                            element_type = "heading"
                        elif max_font_size >= h2_threshold and len(line_text) < 150:
                            heading_level = 2
                            element_type = "heading"
                        elif is_bold and max_font_size >= body_size and len(line_text) < 100:
                            heading_level = 3
                            element_type = "heading"

                        elements.append(
                            ExtractedElement(
                                text=line_text,
                                page_number=page_num,
                                font_size=max_font_size,
                                is_bold=is_bold,
                                heading_level=heading_level,
                                element_type=element_type
                            )
                        )
            doc.close()
            logger.info(f"Successfully extracted {len(elements)} elements from PDF: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            raise e
        return elements
