from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Topic-Aware RAG Question Generator API",
        "version": "1.0.0"
    }
