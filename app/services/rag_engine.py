
"""
Core RAG (Retrieval-Augmented Generation) engine.
Orchestrates retrieval and generation for question answering.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from app.services.bedrock_client import BedrockClient
from app.services.vector_store import VectorStore
from app.services.document_processor import DocumentProcessor
from app.prompts.templates import RAGPromptTemplates
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGEngine:
    """Main RAG engine for question answering over documents."""
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        vector_store: Optional[VectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None
    ):
        """
        Initialize RAG engine.
        
        Args:
            bedrock_client: BedrockClient instance
            vector_store: VectorStore instance
            document_processor: DocumentProcessor instance
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.vector_store = vector_store or VectorStore(self.bedrock_client)
        self.document_processor = document_processor or DocumentProcessor()
        self.prompt_templates = RAGPromptTemplates()
        
        logger.info("Initialized RAGEngine")
    
    def ingest_documents(self, directory: Path) -> Dict[str, Any]:
        """
        Ingest documents from directory into vector store.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting document ingestion from {directory}")
        
        # Process documents
        chunks = self.document_processor.process_directory(directory)
        
        if not chunks:
            logger.warning("No documents were processed")
            return {"status": "error", "message": "No documents found"}
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        # Get statistics
        stats = {
            "status": "success",
            "documents_processed": len(chunks),
            "document_stats": self.document_processor.get_document_stats(chunks),
            "vector_store_stats": self.vector_store.get_stats()
        }
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def query(
        self,
        question: str,
        top_k: int = None,
        temperature: float = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            include_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: '{question[:50]}...'")
        
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(
            query=question,
            k=top_k
        )
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "retrieved_docs": 0
            }
        
        # Step 2: Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {i}] (Relevance: {score:.3f})\n{doc.page_content}\n"
            )
            sources.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "source": doc.metadata.get("source"),
                "file_name": doc.metadata.get("file_name"),
                "score": float(score),
                "preview": doc.page_content[:200] + "..."
            })
        
        context = "\n".join(context_parts)
        
        # Step 3: Generate prompt
        prompt = self.prompt_templates.get_qa_prompt(
            question=question,
            context=context
        )
        
        system_prompt = self.prompt_templates.get_system_prompt()
        
        # Step 4: Generate answer
        try:
            response = self.bedrock_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            answer = response["text"]
            
            # Step 5: Prepare response
            result = {
                "answer": answer,
                "question": question,
                "retrieved_docs": len(retrieved_docs),
                "model": response.get("model_id"),
                "usage": response.get("usage", {})
            }
            
            if include_sources:
                result["sources"] = sources
            
            logger.info(f"Successfully generated answer ({len(answer)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources if include_sources else [],
                "retrieved_docs": len(retrieved_docs),
                "error": str(e)
            }
    
    def save_vector_store(self, path: Optional[Path] = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Optional custom path (uses default if None)
        """
        save_path = path or Path(settings.VECTOR_STORE_PATH)
        self.vector_store.save(save_path)
        logger.info(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, path: Optional[Path] = None) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Optional custom path (uses default if None)
        """
        load_path = path or Path(settings.VECTOR_STORE_PATH)
        self.vector_store.load(load_path)
        logger.info(f"Vector store loaded from {load_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive RAG engine statistics.
        
        Returns:
            Dictionary with all component statistics
        """
        return {
            "vector_store": self.vector_store.get_stats(),
            "configuration": {
                "chunk_size": self.document_processor.chunk_size,
                "chunk_overlap": self.document_processor.chunk_overlap,
                "top_k": settings.TOP_K_RETRIEVAL,
                "model": self.bedrock_client.model_id,
                "embedding_model": self.bedrock_client.embedding_model_id
            }
        }
