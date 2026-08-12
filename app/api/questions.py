from fastapi import APIRouter, HTTPException, status
from app.schemas.question import QuestionGenerationRequest, QuestionGenerationResponse
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaVectorStore
from app.rag.retriever import TopicAwareRetriever
from app.rag.rag_pipeline import RAGPipeline
from app.generation.gemini_client import GeminiClient
from app.generation.question_generator import QuestionGenerator
from app.validation.pipeline import ValidationPipeline
from app.core.logging import logger

router = APIRouter(prefix="/api/v1/questions", tags=["Questions"])

embedding_service = EmbeddingService()
vector_store = ChromaVectorStore()
retriever = TopicAwareRetriever(vector_store=vector_store, embedding_service=embedding_service)
rag_pipeline = RAGPipeline(retriever=retriever)
gemini_client = GeminiClient()
question_generator = QuestionGenerator(gemini_client=gemini_client)
validation_pipeline = ValidationPipeline()


@router.post("/generate", response_model=QuestionGenerationResponse, status_code=status.HTTP_200_OK)
async def generate_questions(request: QuestionGenerationRequest):
    try:
        logger.info(f"Question generation request received for topic '{request.topic}' in chapter '{request.chapter}'")

        # 1. RAG Context Retrieval
        context_str, chunks = rag_pipeline.retrieve_context(request)

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No matching material found for Subject='{request.subject}', Chapter='{request.chapter}', Topic='{request.topic}'."
            )

        all_validated_questions = []
        total_rejected = 0
        original_count = request.count
        max_attempts = 3
        
        for attempt in range(max_attempts):
            needed = original_count - len(all_validated_questions)
            if needed <= 0:
                break
                
            # Ask Gemini to generate more than needed to account for potential rejections
            request.count = needed + 2
            
            # 2. Candidate Question Generation via Gemini
            candidates = question_generator.generate_candidates(request, context_str, chunks)
            
            if not candidates:
                logger.warning(f"Attempt {attempt+1}: Failed to generate candidates.")
                continue

            # Restore the 'needed' count for the validation pipeline so it knows when to stop
            request.count = needed
            
            # 3. Quality, Grounding, Topic & Duplicate Validation Pipeline
            batch_validated, batch_rejected = validation_pipeline.process(candidates, request, chunks)
            total_rejected += batch_rejected
            
            # Ensure we don't add duplicates from previous attempts
            for vq in batch_validated:
                is_dup = False
                for existing in all_validated_questions:
                    cand_tokens = set(vq.question.lower().split())
                    acc_tokens = set(existing.question.lower().split())
                    if cand_tokens and acc_tokens:
                        intersection = cand_tokens.intersection(acc_tokens)
                        union = cand_tokens.union(acc_tokens)
                        if union and len(intersection) / len(union) >= 0.85:
                            is_dup = True
                            break
                if is_dup:
                    total_rejected += 1
                else:
                    all_validated_questions.append(vq)
                    if len(all_validated_questions) >= original_count:
                        break

        # Restore original count for response
        request.count = original_count

        if not all_validated_questions:
            raise HTTPException(
                status_code=422,
                detail="All generated candidate questions failed quality or groundedness validation checks after multiple attempts."
            )

        return QuestionGenerationResponse(
            questions=all_validated_questions,
            total_generated=len(all_validated_questions),
            total_requested=request.count,
            rejected_count=total_rejected,
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            subtopic=request.subtopic
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal question generation error: {str(e)}")
