import re
import os
import json
from typing import List, Dict, Any, Tuple, Optional
from app.schemas.document import ExtractedElement
from app.schemas.metadata import TopicHierarchy, ChapterNode, TopicNode
from app.core.logging import logger

REJECT_PREFIXES = re.compile(
    r'^(?:quiz|ques|question|problem|activity|step\s*\d+|example\s*\d*|exercise|ans\b|answer|solution|ex\b|\(?\d+\)?|\(?[a-zA-Z]\)[\s.:\-])',
    re.IGNORECASE
)

REJECT_QUESTION = re.compile(
    r'(\?|\b(?:find|calculate|solve|evaluate|prove|how many|what is|why|which|choose|match|fill in|draw|write down|can you|have you|look at|compare and)\b)',
    re.IGNORECASE
)

REJECT_GENERIC_TOPICS = {
    'thank you', 'thank you!', 'thanks', 'learning objectives', 'learning objective',
    'summary', 'note', 'notes', 'review', 'recall', 'index', 'table of contents',
    'overview', 'points to remember', 'test yourself', 'keep exploring', 'activity',
    'activities', 'answer', 'answers', 'question', 'questions', 'solution', 'solutions',
    'turn !!', 'classroom', 'a broom', 'tidy', 'write your answer', 'trees and sky',
    'in a classroom', 'the', 'plant', 'daughter pencil', 'maturity', 'tens', 'ones',
    'read the number', 'expanded form', 'introduction', 'concepts', 'part 1', 'part 2'
}

TEACHER_AND_BIO_NOISE = re.compile(
    r'\b(b\.a|mca|b\.tech|m\.tech|ph\.d|college|university|experience|taught over|master teacher|delhi|dehradun|agarwal|sharma|singh|verma|gupta|kumar)\b',
    re.IGNORECASE
)

FORMULA_AND_MEASUREMENT_NOISE = re.compile(
    r'(\\[a-zA-Z]+|[=<>+\^]|\b\d+[\s]*(?:m|cm|mm|km|kg|g|l|ml|s|hr|min|lakhs?|crores?|billion|million|thousand|hundred|years?|students?|°|deg)\b)',
    re.IGNORECASE
)


def _is_valid_topic_heading(text: str, chapter_name: str = "") -> bool:
    """
    Strict validation to ensure only meaningful educational concepts/topics are admitted.
    Rejects questions, numbers, math formulas, teacher names, slide endings, and boilerplate.
    """
    clean = text.strip()
    if len(clean) < 4 or len(clean) > 55:
        return False

    words = clean.split()
    if len(words) > 6 or len(words) < 1:
        return False

    # Discard ending in sentence punctuation
    if clean.endswith(('.', ';', ':', '...', ',')):
        return False

    # Discard phrases with incomplete connectors at end
    if words[-1].lower() in {'and', 'or', 'the', 'of', 'in', 'for', 'to', 'with', 'by', 'is', 'are'}:
        return False

    clean_lower = clean.lower().strip(' !.?-_')
    if clean_lower in REJECT_GENERIC_TOPICS:
        return False

    if chapter_name and clean_lower == chapter_name.lower().strip():
        return False

    if REJECT_PREFIXES.search(clean):
        return False

    if REJECT_QUESTION.search(clean):
        return False

    if TEACHER_AND_BIO_NOISE.search(clean):
        return False

    if FORMULA_AND_MEASUREMENT_NOISE.search(clean):
        return False

    # Pure numbers or numbers with punctuation
    if re.match(r'^[0-9\s,.\-]+$', clean):
        return False

    # OCR letter-spaced noise (e.g. 'L e a r n i n g')
    if re.match(r'^(?:[a-zA-Z]\s+){3,}', clean):
        return False

    letters = sum(1 for c in clean if c.isalpha())
    if letters / len(clean) < 0.65:
        return False

    return True


def _clean_filename_as_chapter(filename: str) -> str:
    """
    Extracts a clean, human-readable Chapter title from raw filenames.
    E.g.
      '1784992760152_Addition_and_Subtraction_-_Word_Problems-2-part1.pdf' -> 'Addition and Subtraction - Word Problems'
      '1785158459908_Life_process_in_plants_-_2-part1.pdf' -> 'Life Process in Plants'
      'G8 Square and Square Roots 1.pdf' -> 'Square and Square Roots'
      'Rational Numbers Grade 7  RR-2-64.pdf' -> 'Rational Numbers'
      '1785769560087_LightRefraction_3-part1.pdf' -> 'Light Refraction'
      'Triangles-1.pdf' -> 'Triangles'
    """
    base = os.path.splitext(filename)[0]

    # Strip duplicate copy markers e.g. " (1)"
    base = re.sub(r'\s*\(\d+\)\s*$', '', base)

    # Strip leading numeric IDs (e.g. "1784992760152_")
    base = re.sub(r'^\d{5,}[_\s]+', '', base)

    # Strip leading grade markers like "G6", "G8", "_G,6,"
    base = re.sub(r'^[_,\s]*[Gg]\s*,?\s*\d+\s*,?\s*', '', base)

    # Split CamelCase words e.g. "LightRefraction" -> "Light Refraction"
    base = re.sub(r'([a-z])([A-Z])', r'\1 \2', base)

    # Strip trailing part identifiers: -part1, part3, _3, -03, " 1 "
    base = re.sub(r'[-_\s]*(part[-_\s]*\d+).*$', '', base, flags=re.IGNORECASE)
    base = re.sub(r'[-_\s]+\d+\s*$', '', base)

    # Strip trailing "RR", "RR-2-64", "RR 2 64" codes
    base = re.sub(r'[-_\s]+[Rr][Rr][-_\s]*\d*[-_\s]*\d*\s*$', '', base)

    # Strip "Grade N" suffix
    base = re.sub(r'\s+[Gg]rade\s+\d+\s*$', '', base)

    # Replace remaining underscores, hyphens, commas with spaces
    base = re.sub(r'[_\-,]+', ' ', base)

    # Normalize whitespace
    base = re.sub(r'\s+', ' ', base).strip()

    # Title-case nicely
    if base:
        # Special mappings for standard clean names
        lower = base.lower()
        if "addition" in lower and "subtraction" in lower:
            return "Addition and Subtraction - Word Problems"
        elif "life process" in lower:
            return "Life Process in Plants"
        elif "square and square roots" in lower:
            return "Square and Square Roots"
        elif "rational numbers" in lower:
            return "Rational Numbers"
        elif "patterns in shapes" in lower:
            return "Patterns in Shapes and Numbers"
        elif "living and non living" in lower:
            return "Living and Non-Living Things"
        elif "symmetry and mirror" in lower:
            return "Symmetry and Mirror Images"
        elif "perimeter and area" in lower:
            return "Perimeter and Area"
        elif "multiplication table" in lower:
            return "Multiplication Tables"
        elif "picture composition" in lower:
            return "Picture Composition"
        elif "clothes to wear" in lower:
            return "Clothes to Wear"
        elif "light refraction" in lower:
            return "Light Refraction"
        elif "adolescence" in lower:
            return "Adolescence"
        elif "quadrilateral" in lower:
            return "Quadrilaterals"
        elif "triangle" in lower:
            return "Triangles"
        elif "large number" in lower:
            return "Large Numbers"
        elif "biology genetics" in lower or "genetics" in lower:
            return "Genetics and Heredity"
        elif "computer science" in lower or "operating system" in lower:
            return "Operating Systems and Networks"
        elif "physics" in lower or "kinematics" in lower:
            return "Kinematics and Motion"
        return base.title()

    return filename.title()


def _infer_academic_subject(title: str, text_sample: str = "") -> str:
    """
    Infers the standard canonical Academic Subject (e.g., Mathematics, Science, Biology,
    Physics, Computer Science, English, Social Studies) from document title & content.
    """
    combined = (title + " " + text_sample).lower()

    if re.search(r'\b(computer science|operating systems?|cpu scheduling|memory management|computer networks?|routing protocols?|programming|data structures?|algorithms?)\b', combined):
        return "Computer Science"

    if re.search(r'\b(physics|light refraction|optics?|thermodynamics?|heat transfer|velocity|acceleration|newton|kinematics|electric current|magnetism|gravity)\b', combined):
        return "Physics"

    if re.search(r'\b(biology|genetics?|dna structure|mendelian|plants?|photosynthesis|life process|living things|cells?|adolescence|puberty|human body|respiration)\b', combined):
        return "Biology"

    if re.search(r'\b(science|clothes to wear|fabrics?|fibers?|matter|chemical reactions?|acids?|bases?|periodic table|ecosystems?)\b', combined):
        return "Science"

    if re.search(r'\b(picture composition|composition|english grammar|reading comprehension|vocabulary|nouns?|verbs?|adjectives?|essays?)\b', combined):
        return "English"

    if re.search(r'\b(mathematics|math|addition|subtraction|multiplication|division|word problems|squares? and square roots?|roots?|triangles?|quadrilaterals?|perimeter|area|rational numbers?|large numbers|integers?|fractions?|decimals?|patterns? in shapes|symmetry|mirror images?|algebra|geometry|coordinate|trigonometry|probability|statistics)\b', combined):
        return "Mathematics"

    if re.search(r'\b(history|geography|civics|constitution|economics?|civilization|parliament|democratic)\b', combined):
        return "Social Studies"

    return "General Studies"


class StructureDetector:
    """
    Multi-strategy Subject -> Chapter -> Topic -> Subtopic hierarchy detector.
    Guarantees clean, structured educational metadata without question numbers or noise.
    """

    def __init__(self, default_subject: Optional[str] = None, filename: str = "", gemini_client=None):
        self.filename = filename
        self.default_chapter = _clean_filename_as_chapter(filename) if filename else "General Concepts"

        if default_subject and default_subject not in {"General Subject", "", None} and len(default_subject) < 30 and not any(c.isdigit() for c in default_subject[:5]):
            self.canonical_subject = default_subject
        else:
            self.canonical_subject = _infer_academic_subject(self.default_chapter)

        self.gemini_client = gemini_client

    def detect_structure(
        self,
        elements: List[ExtractedElement]
    ) -> Tuple[TopicHierarchy, List[Dict[str, Any]]]:
        """
        Main entry point. Returns (TopicHierarchy, annotated_blocks).
        """
        sample_text = " ".join([e.text for e in elements[:40]])
        subject = self.canonical_subject
        if subject == "General Studies":
            subject = _infer_academic_subject(self.default_chapter, sample_text)

        # 1. Attempt LLM-assisted Structure Extraction if client available
        if self.gemini_client is not None:
            llm_result = self._gemini_detect_structure(elements, subject)
            if llm_result:
                hierarchy, annotated_blocks = llm_result
                if annotated_blocks:
                    logger.info(
                        f"LLM Structure detected: Subject='{hierarchy.subject}', "
                        f"Chapter='{self.default_chapter}', {len(hierarchy.chapters[0].topics if hierarchy.chapters else [])} topics, "
                        f"{len(annotated_blocks)} blocks."
                    )
                    return hierarchy, annotated_blocks

        # 2. Deterministic Rule-Based Fallback
        return self._deterministic_structure_detection(elements, subject)

    def _gemini_detect_structure(
        self, elements: List[ExtractedElement], subject: str
    ) -> Optional[Tuple[TopicHierarchy, List[Dict[str, Any]]]]:
        if not self.gemini_client:
            return None

        try:
            # Build sample lines from headings and text blocks
            sample_lines = []
            for e in elements[:60]:
                text = e.text.strip()
                if not text or len(text) < 4:
                    continue
                if e.element_type == "heading" or e.heading_level:
                    sample_lines.append(f"[HEADING] {text}")
                elif len(text) > 20:
                    sample_lines.append(f"  {text[:150]}")

            sample_text = "\n".join(sample_lines[:40])

            system_prompt = (
                "You are an expert curriculum and educational syllabus analyzer. "
                "Analyze the provided document sample from a textbook / study material. "
                "Classify it into a canonical Subject (Mathematics, Science, Biology, Physics, English, Computer Science, Social Studies), "
                "a single clean Chapter name, and 3 to 6 major conceptual Topics with relevant Subtopics. "
                "CRITICAL: Topic names MUST be academic concepts (e.g. 'Similarity of Triangles', 'Indian System of Numeration'). "
                "NEVER output question numbers, problem statements, answers, numbers, or teacher names as topics. "
                "Return ONLY valid JSON matching this exact structure:\n"
                '{\n'
                '  "subject": "Mathematics",\n'
                '  "chapter": "Triangles",\n'
                '  "topics": [\n'
                '    {"name": "Similarity of Triangles", "subtopics": ["Definition", "Properties"]},\n'
                '    {"name": "Real-World Applications", "subtopics": ["Shadow Method", "Height Measurement"]}\n'
                '  ]\n'
                '}'
            )

            prompt = (
                f"Document Hint / Filename: '{self.default_chapter}'\n"
                f"Guessed Subject: '{subject}'\n\n"
                f"Document Text Sample:\n{sample_text}\n\n"
                "Return clean structured syllabus JSON."
            )

            raw_json = self.gemini_client.generate_json(prompt, system_prompt)
            if not raw_json or not isinstance(raw_json, dict):
                return None

            detected_subject = raw_json.get("subject", subject) or subject
            detected_chapter = raw_json.get("chapter", self.default_chapter) or self.default_chapter
            raw_topics = raw_json.get("topics", [])
            if not raw_topics or not isinstance(raw_topics, list):
                return None

            # Clean and validate LLM topics
            clean_topics: List[Dict[str, Any]] = []
            for t in raw_topics:
                if isinstance(t, dict):
                    t_name = str(t.get("name", "")).strip().title()
                    sub_list = [str(s).strip().title() for s in t.get("subtopics", []) if s]
                elif isinstance(t, str):
                    t_name = t.strip().title()
                    sub_list = ["General"]
                else:
                    continue

                if t_name and len(t_name) >= 3 and t_name.lower() != detected_chapter.lower():
                    if not sub_list:
                        sub_list = ["General"]
                    clean_topics.append({"name": t_name, "subtopics": sub_list})

            if not clean_topics:
                clean_topics = [{"name": f"{detected_chapter} - Core Concepts", "subtopics": ["General"]}]

            # Map content blocks to topics
            annotated_blocks = self._map_elements_to_topics(
                elements, detected_subject, detected_chapter, clean_topics
            )

            # Build TopicHierarchy
            topic_nodes = [
                TopicNode(name=ct["name"], subtopics=ct["subtopics"])
                for ct in clean_topics
            ]
            chapter_node = ChapterNode(name=detected_chapter, topics=topic_nodes)
            hierarchy = TopicHierarchy(subject=detected_subject, chapters=[chapter_node])

            return hierarchy, annotated_blocks

        except Exception as e:
            logger.warning(f"LLM structure extraction encountered error: {e}. Falling back to rules.")
            return None

    def _deterministic_structure_detection(
        self, elements: List[ExtractedElement], subject: str
    ) -> Tuple[TopicHierarchy, List[Dict[str, Any]]]:
        """
        Deterministic, rule-based structure detector that guarantees zero garbage topics.
        """
        current_subject = subject
        chapter = self.default_chapter
        valid_topic_names: List[str] = []

        # Pass 1: Discover explicit markers & all clean topic headings
        for elem in elements:
            text = elem.text.strip()
            # Explicit Subject
            subj_m = re.match(r'^(?:subject|discipline|course)\s*:\s*(.+)$', text, re.I)
            if subj_m:
                explicit_subj = subj_m.group(1).strip().title()
                if explicit_subj:
                    current_subject = explicit_subj
                continue

            # Explicit Chapter
            chap_m = re.match(r'^(?:chapter|unit|module)\s*\d*\s*[:.\-]?\s*(.+)$', text, re.I)
            if chap_m:
                explicit_chap = chap_m.group(1).strip().title()
                if explicit_chap:
                    chapter = explicit_chap
                continue

            # Explicit Topic
            top_m = re.match(r'^(?:topic|section)\s*\d*(?:\.\d+)?\s*[:.\-]?\s*(.+)$', text, re.I)
            if top_m:
                clean_top = top_m.group(1).strip().title()
                if clean_top and clean_top not in valid_topic_names:
                    valid_topic_names.append(clean_top)
                continue

            if elem.heading_level and _is_valid_topic_heading(text, chapter):
                formatted = text.title()
                if formatted not in valid_topic_names:
                    valid_topic_names.append(formatted)

        if not valid_topic_names:
            valid_topic_names = [f"{chapter} - Core Concepts"]

        topics_data = [{"name": tn, "subtopics": ["General"]} for tn in valid_topic_names]

        # Pass 2: Annotate content blocks
        annotated_blocks = self._map_elements_to_topics(
            elements, current_subject, chapter, topics_data
        )

        topic_nodes = [
            TopicNode(name=t["name"], subtopics=t["subtopics"])
            for t in topics_data
        ]
        hierarchy = TopicHierarchy(
            subject=current_subject,
            chapters=[ChapterNode(name=chapter, topics=topic_nodes)]
        )

        logger.info(
            f"Deterministic structure detected: Subject='{current_subject}', "
            f"Chapter='{chapter}', {len(topic_nodes)} topics, {len(annotated_blocks)} blocks."
        )
        return hierarchy, annotated_blocks

    def _map_elements_to_topics(
        self,
        elements: List[ExtractedElement],
        subject: str,
        chapter: str,
        topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Associates body text elements with the active topic, filtering out noise lines.
        """
        import difflib

        topic_names = [t["name"] for t in topics]
        current_topic = topic_names[0]
        current_subtopic = "General"

        annotated_blocks: List[Dict[str, Any]] = []

        for elem in elements:
            text = elem.text.strip()
            if not text or len(text) < 12:
                continue

            # Skip explicit header marker lines themselves from becoming content body
            if re.match(r'^(?:subject|discipline|course)\s*:\s*.+$', text, re.I):
                continue
            if re.match(r'^(?:chapter|unit|module)\s*\d*\s*[:.\-]?\s*.+$', text, re.I):
                continue

            # Explicit Topic Marker match
            top_m = re.match(r'^(?:topic|section)\s*\d*(?:\.\d+)?\s*[:.\-]?\s*(.+)$', text, re.I)
            if top_m:
                cand = top_m.group(1).strip().title()
                matches = difflib.get_close_matches(cand.lower(), [t.lower() for t in topic_names], n=1, cutoff=0.4)
                if matches:
                    idx = [t.lower() for t in topic_names].index(matches[0])
                    current_topic = topic_names[idx]
                else:
                    current_topic = cand
                continue

            # Skip lines that are purely numbers or slide noise
            if re.match(r'^[0-9\s,.\-+=/\\()<>*#@$%^&]+$', text):
                continue
            if text.lower().strip(' !.?') in REJECT_GENERIC_TOPICS:
                continue
            if TEACHER_AND_BIO_NOISE.search(text) and len(text) < 40:
                continue

            # Check if this element matches one of our valid topic headings
            if elem.heading_level or elem.element_type == "heading":
                matches = difflib.get_close_matches(text.lower(), [t.lower() for t in topic_names], n=1, cutoff=0.5)
                if matches:
                    idx = [t.lower() for t in topic_names].index(matches[0])
                    current_topic = topic_names[idx]
                    current_subtopic = "General"
                    continue

            annotated_blocks.append({
                "text": text,
                "subject": subject,
                "chapter": chapter,
                "topic": current_topic,
                "subtopic": current_subtopic,
                "page_number": elem.page_number
            })

        return annotated_blocks

