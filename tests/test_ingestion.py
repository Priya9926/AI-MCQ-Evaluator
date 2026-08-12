import os
import pytest
from app.ingestion.txt_parser import TXTParser
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.docx_parser import DOCXParser
from app.ingestion.text_cleaner import TextCleaner
from app.ingestion.structure_detector import StructureDetector
from app.ingestion.folder_ingestor import FolderIngestor
from app.vectorstore.chroma_store import ChromaVectorStore
from app.embeddings.embedding_service import EmbeddingService


def test_txt_parser(tmp_path):
    txt_file = tmp_path / "biology_sample.txt"
    content = (
        "Subject: Biology\n"
        "Chapter 1: Cell Biology\n"
        "Topic: Mitochondria\n"
        "Mitochondria are membrane-bound cell organelles that generate chemical energy needed to power biochemical reactions."
    )
    txt_file.write_text(content, encoding="utf-8")

    parser = TXTParser()
    elements = parser.parse(str(txt_file))
    assert len(elements) == 4
    assert elements[0].text == "Subject: Biology"

    cleaner = TextCleaner()
    cleaned = cleaner.clean_elements(elements)
    assert len(cleaned) == 4

    detector = StructureDetector()
    hierarchy, blocks = detector.detect_structure(cleaned)
    assert hierarchy.subject == "Biology"
    assert len(blocks) > 0
    assert blocks[0]["topic"] == "Mitochondria"


def test_folder_ingestor(tmp_path):
    upload_folder = tmp_path / "custom_upload"
    upload_folder.mkdir()
    chroma_dir = tmp_path / "chroma_test"

    # Create an arbitrary Computer Science document
    cs_file = upload_folder / "computer_science.txt"
    cs_content = (
        "Subject: Computer Science\n\n"
        "Chapter: Operating Systems\n\n"
        "Topic: Memory Management\n"
        "Memory management is the functionality of an OS which handles or manages primary memory and moves processes back and forth between main memory and disk during execution.\n\n"
        "Topic: CPU Scheduling\n"
        "CPU scheduling is the basis of multiprogrammed operating systems. By switching the CPU among processes, the operating system can make the computer more productive.\n"
    )
    cs_file.write_text(cs_content, encoding="utf-8")

    vector_store = ChromaVectorStore(collection_name="test_folder_col", persist_directory=str(chroma_dir))
    embedding_service = EmbeddingService()
    ingestor = FolderIngestor(vector_store=vector_store, embedding_service=embedding_service)

    results = ingestor.ingest_all_from_folder(str(upload_folder))
    assert len(results) == 1
    assert results[0].filename == "computer_science.txt"
    assert results[0].detected_hierarchy.subject == "Computer Science"
    assert results[0].total_chunks >= 2
