import os
import sys
from app.ingestion.folder_ingestor import FolderIngestor
from app.core.config import settings
from app.core.logging import logger


def main():
    upload_dir = settings.UPLOADS_DIR
    print("=" * 60)
    print("      Topic-Aware RAG Batch Ingestor")
    print(f"      Scanning directory: {os.path.abspath(upload_dir)}")
    print("=" * 60)

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
        print(f"Created folder '{upload_dir}'. Place your PDF, DOCX, or TXT files there.")
        return

    ingestor = FolderIngestor()
    results = ingestor.ingest_all_from_folder(upload_dir)

    if not results:
        print(f"\nNo eligible documents (.pdf, .docx, .txt) found in '{upload_dir}'.")
        print("Drop your study materials into the 'upload' folder and run this script again.")
        return

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"\nSuccessfully ingested {len(results)} document(s):")
    for res in results:
        print(f"\n* Document: {res.filename} (ID: {res.document_id})")
        print(f"   Subject: {res.detected_hierarchy.subject}")
        print(f"   Total Pages: {res.total_pages}")
        print(f"   Total Chunks: {res.total_chunks}")
        print("   Detected Hierarchy:")
        for chap in res.detected_hierarchy.chapters:
            print(f"     |-- Chapter: {chap.name}")
            for top in chap.topics:
                print(f"           |-- Topic: {top.name}")
                for subtop in top.subtopics:
                    if subtop != "General":
                        print(f"                 |-- Subtopic: {subtop}")

    print("\n" + "=" * 60)
    print("All materials indexed and ready for topic-aware question generation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
