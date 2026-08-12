import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api import upload, documents, questions, health
from app.core.config import settings
from app.core.logging import logger

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML_PATH = os.path.join(STATIC_DIR, "index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Topic-Aware RAG Question Generator API starting up...")
    yield
    logger.info("Topic-Aware RAG Question Generator API shutting down.")


app = FastAPI(
    title="Topic-Aware RAG Question Generator",
    description="Production-Grade RAG Pipeline to generate topic-isolated, grounded assessment questions strictly from uploaded educational material.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(questions.router)
app.include_router(health.router)


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def root():
    """Serves the interactive Topic-Aware RAG Question Generator Web Dashboard."""
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Topic-Aware RAG Question Generator</h1><p>Visit <a href='/docs'>/docs</a> for Swagger API.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
