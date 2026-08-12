# Topic-Aware RAG Question Generator

A production-grade, topic-isolated RAG (Retrieval-Augmented Generation) pipeline for educational assessment question generation. Built with **FastAPI**, **Gemini API (`google-genai` SDK)**, **ChromaDB**, and **Pydantic** to ingest educational materials (PDF, DOCX, TXT), detect Subject → Chapter → Topic hierarchy, and generate strictly grounded, validated multiple-choice assessment questions.

---

## 🏗 Architecture & Data Flow

```text
Study Materials (PDF / DOCX / TXT)
       ↓
Document Parser (PyMuPDF / python-docx)
       ↓
Text Cleaning & De-spacing
       ↓
Structure & Academic Subject Detection (Subject → Chapter → Topic → Subtopic)
       ↓
Topic-Aware Semantic Chunking (Strict topic boundary isolation)
       ↓
Metadata Builder + Embeddings Generation
       ↓
ChromaDB Vector Store (Indexed with rich metadata filters)
       ↓
User Question Request (Subject, Chapter, Topic, Count, Difficulty)
       ↓
Metadata-Priority Filtering + Vector Similarity Search
       ↓
Retrieved Topic Context (Zero cross-topic leakage)
       ↓
Gemini LLM Question Generator (Strict ground-truth prompting)
       ↓
Multi-Stage Validation Pipeline:
  ├── Schema & Answer Validator
  ├── Topic Consistency Check
  ├── Groundedness Verifier (Mandatory source chunk traceability)
  └── Semantic Duplicate Detector
       ↓
Final Grounded Assessment Questions (MCQs with Explanations)
```

---

## 🛠 Technology Stack

* **Backend API:** FastAPI + Uvicorn
* **LLM Engine:** Google Gemini (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-flash-latest`) via modern `google-genai` SDK
* **Vector Store:** ChromaDB with modular abstraction (`VectorStoreBase`)
* **Document Extraction:** PyMuPDF (`fitz`), `python-docx`
* **Data Validation:** Pydantic v2
* **Web UI:** Interactive Web Dashboard at `http://127.0.0.1:8000/`

---

## 📂 Project Structure

```text
app/
├── main.py                     # FastAPI application & Web Dashboard route
├── api/
│   ├── upload.py               # Document upload & folder ingestion API
│   ├── documents.py            # Topic hierarchy catalog & stats API
│   ├── questions.py            # Topic-aware question generation API
│   └── health.py               # Health probe endpoint
├── core/
│   ├── config.py               # Centralized settings & .env parser
│   └── logging.py              # Structured application logger
├── ingestion/
│   ├── pdf_parser.py           # Adaptive font-size threshold PDF parser
│   ├── docx_parser.py          # Word document structural parser
│   ├── txt_parser.py           # Plain text extractor
│   ├── text_cleaner.py         # De-spacing & text normalizer
│   ├── structure_detector.py   # Multi-strategy Subject/Chapter/Topic classifier
│   ├── topic_chunker.py        # Semantic chunker enforcing topic boundaries
│   ├── folder_ingestor.py      # Batch folder ingestion engine
│   └── metadata_builder.py     # Chunk metadata constructor
├── embeddings/
│   └── embedding_service.py    # Text & query vector embedding service
├── vectorstore/
│   ├── base.py                 # Abstract VectorStoreBase interface
│   └── chroma_store.py         # ChromaDB implementation with $and filter logic
├── rag/
│   ├── retriever.py            # TopicAwareRetriever (Metadata filter before vector search)
│   ├── context_builder.py      # Grounded context assembler with chunk citations
│   └── rag_pipeline.py         # Orchestration pipeline
├── generation/
│   ├── gemini_client.py        # Gemini SDK client with auto-model fallback
│   ├── prompt_builder.py       # Strict grounding prompt builder
│   └── question_generator.py   # Candidate question generation engine
├── validation/
│   ├── answer_validator.py     # Option validity & single correct key check
│   ├── topic_validator.py      # Topic match & boundary validator
│   ├── groundedness.py         # Source context evidence verifier
│   ├── duplicate_detector.py   # Semantic similarity duplicate remover
│   └── pipeline.py             # Sequential validation orchestrator
├── schemas/
│   ├── document.py             # Chunk & extracted element schemas
│   ├── metadata.py             # Hierarchy & chunk metadata schemas
│   ├── question.py             # MCQ request & response schemas
│   └── generation.py           # Raw LLM response schemas
├── static/
│   └── index.html              # Interactive dark-mode Web Dashboard
└── utils/
    └── helpers.py              # ID generator & text utilities

upload/                         # Directory for uploading batch study materials
data/
└── chroma/                     # Persistent ChromaDB vector database

tests/
├── test_ingestion.py           # Document parser & folder ingestion tests
├── test_chunking.py            # Topic isolation chunking tests
├── test_retrieval.py           # Topic isolation retrieval tests
├── test_generation.py          # Question generator tests
├── test_validation.py          # Validation pipeline tests
└── test_end_to_end.py          # Full end-to-end question generation test
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create or edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
EMBEDDING_MODEL=gemini-embedding-001
CHROMA_PATH=./data/chroma
UPLOADS_DIR=./upload
PROCESSED_DIR=./data/processed
CHUNK_SIZE=200
CHUNK_OVERLAP=50
TOP_K=8
DUPLICATE_SIMILARITY_THRESHOLD=0.85
```

### 3. Ingest Study Materials
Place your educational files (`.pdf`, `.docx`, `.txt`) into the `./upload` folder, then run:
```bash
python ingest_folder.py
```

### 4. Start the Application
```bash
uvicorn app.main:app --reload
```

* **Interactive Web Dashboard:** Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
* **Interactive API Documentation:** Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest -v
```

All 10 tests verify:
1. Topic boundary semantic chunking isolation
2. End-to-end question generation with zero cross-topic leakage
3. Document extraction (PDF, DOCX, TXT)
4. Vector metadata filtering (`$and` clauses)
5. Answer structure validation
6. Groundedness validation with mandatory `source_chunk_ids`
7. Semantic duplicate question detection
8. Interactive Web UI and catalog endpoints
