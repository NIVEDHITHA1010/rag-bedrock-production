
"""
FastAPI application for RAG system API.
Provides REST endpoints for document processing and querying.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.services.rag_engine import RAGEngine
from app.config import get_settings
from app.utils.logger import get_logger

# Initialize
logger = get_logger(__name__)
settings = get_settings()
app = FastAPI(
    title="Advanced RAG API",
    description="Production-grade RAG system with AWS Bedrock",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

# Try to load existing vector store
try:
    vector_store_path = Path(settings.VECTOR_STORE_PATH)
    if vector_store_path.exists():
        rag_engine.load_vector_store()
        logger.info("Loaded existing vector store")
except Exception as e:
    logger.warning(f"Could not load vector store: {e}")


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    temperature: Optional[float] = Field(0.1, description="LLM temperature")
    include_sources: bool = Field(True, description="Include source documents")


class QueryResponse(BaseModel):
    answer: str
    question: str
    retrieved_docs: int
    sources: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    vector_store_docs: int
    configuration: Dict[str, Any]


class IngestRequest(BaseModel):
    directory_path: str = Field(..., description="Path to documents directory")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    message: str


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Advanced RAG API with AWS Bedrock",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "ingest": "/ingest",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and configuration.
    
    Returns:
        System health status and statistics
    """
    try:
        stats = rag_engine.get_stats()
        return HealthResponse(
            status="healthy",
            vector_store_docs=stats["vector_store"]["total_documents"],
            configuration=stats["configuration"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Generated answer with sources and metadata
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            include_sources=request.include_sources
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from a directory into the vector store.
    
    Args:
        request: Ingest request with directory path
        
    Returns:
        Ingestion status and statistics
    """
    try:
        directory = Path(request.directory_path)
        
        if not directory.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {request.directory_path}"
            )
        
        if not directory.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a directory: {request.directory_path}"
            )
        
        result = rag_engine.ingest_documents(directory)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Ingestion failed")
            )
        
        # Save updated vector store
        rag_engine.save_vector_store()
        
        return IngestResponse(
            status="success",
            documents_processed=result["documents_processed"],
            message=f"Successfully ingested {result['documents_processed']} document chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@app.get("/stats", tags=["System"])
async def get_statistics():
    """
    Get comprehensive system statistics.
    
    Returns:
        Detailed statistics about the RAG system
    """
    try:
        return rag_engine.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.delete("/vector-store", tags=["System"])
async def clear_vector_store():
    """
    Clear all documents from the vector store.
    
    Returns:
        Confirmation message
    """
    try:
        rag_engine.vector_store.delete_all()
        return {"status": "success", "message": "Vector store cleared"}
    except Exception as e:
        logger.error(f"Failed to clear vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear vector store: {str(e)}"
        )


# Run application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
