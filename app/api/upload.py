import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.document import DocumentUploadResponse
from app.ingestion.folder_ingestor import FolderIngestor
from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
folder_ingestor = FolderIngestor()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a single document (PDF, DOCX, TXT) to the upload folder and processes it into the RAG vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ['.pdf', '.docx', '.txt']:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Must be PDF, DOCX, or TXT.")

    save_dir = settings.UPLOADS_DIR
    os.makedirs(save_dir, exist_ok=True)
    saved_path = os.path.join(save_dir, filename)

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {saved_path}")

        response = folder_ingestor.ingest_file(saved_path)
        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Internal document processing error: {str(e)}")


@router.post("/ingest-folder", response_model=List[DocumentUploadResponse], status_code=status.HTTP_200_OK)
async def ingest_folder():
    """Scans the `upload/` folder and ingests all pending documents into the topic-aware RAG vector store."""
    try:
        results = folder_ingestor.ingest_all_from_folder()
        return results
    except Exception as e:
        logger.error(f"Error ingesting folder: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting upload folder: {str(e)}")


@router.post("/reset-and-reindex", response_model=List[DocumentUploadResponse], status_code=status.HTTP_200_OK)
async def reset_and_reindex():
    """Purges all existing records in the vector database and completely re-indexes the upload folder from scratch."""
    try:
        folder_ingestor.vector_store.clear()
        results = folder_ingestor.ingest_all_from_folder()
        return results
    except Exception as e:
        logger.error(f"Error during reset and reindex: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting and reindexing: {str(e)}")
