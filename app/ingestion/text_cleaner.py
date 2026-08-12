import re
from typing import List
from app.schemas.document import ExtractedElement


def _clean_spaced_letters(text: str) -> str:
    """
    Cleans wide-spaced text extracted from PDFs with unusual kerning.
    E.g. 'L e a r n i n g   o b j e c t i v e s' -> 'Learning objectives'
         'R e s p i r a t i o n' -> 'Respiration'
    """
    def merge_run(m):
        return m.group(0).replace(" ", "")

    # Runs of 3 or more single letters separated by single space (e.g. "R e s p i r a t i o n")
    cleaned = re.sub(r'(?:(?<=\s)|(?<=^))(?:[A-Za-z]\s){2,}[A-Za-z](?=\s|$|[.,:;!?])', merge_run, text)
    return cleaned


class TextCleaner:
    """Cleans whitespace, removes headers/footers, and normalizes extracted elements."""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # 1. Fix wide-spaced letters
        cleaned = _clean_spaced_letters(text)
        # 2. Replace multiple spaces/newlines with single space
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n+', '\n', cleaned)
        return cleaned.strip()

    def clean_elements(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        cleaned_elements: List[ExtractedElement] = []
        for elem in elements:
            cleaned_str = self.clean_text(elem.text)
            if not cleaned_str:
                continue

            # Filter typical page header/footer patterns like "Page 1 of 10" or "All Rights Reserved"
            if re.match(r'^page\s+\d+(\s+of\s+\d+)?$', cleaned_str, re.IGNORECASE):
                continue
            if re.match(r'^copyright\s+.*', cleaned_str, re.IGNORECASE):
                continue

            elem.text = cleaned_str
            cleaned_elements.append(elem)

        return cleaned_elements
