from fastapi import APIRouter, HTTPException
from app.schemas.metadata import TopicHierarchy
from app.vectorstore.chroma_store import ChromaVectorStore
from app.core.logging import logger

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
vector_store = ChromaVectorStore()


@router.get("/topics/all", response_model=list[TopicHierarchy])
async def get_all_topics():
    """Returns the complete topic hierarchy across all indexed subjects."""
    try:
        hierarchies = vector_store.get_all_hierarchies()
        return hierarchies
    except Exception as e:
        logger.error(f"Error fetching all topic hierarchies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_document_stats():
    """Returns total chunks, unique subjects, chapters, topics and document counts."""
    return vector_store.get_stats()


@router.get("/{document_id}/topics", response_model=TopicHierarchy)
async def get_document_topics(document_id: str):
    try:
        hierarchy = vector_store.get_topics_hierarchy(source_document=document_id)
        if not hierarchy.chapters:
            raise HTTPException(status_code=404, detail=f"Document ID '{document_id}' not found or has no topics.")
        return hierarchy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching topics for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

