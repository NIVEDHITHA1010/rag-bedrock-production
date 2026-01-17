"""
Document processing pipeline for loading and chunking documents.
Supports multiple file formats and optimized text splitting.
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Handle document loading, parsing, and chunking."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(
            f"Initialized DocumentProcessor with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document based on file type.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file format not supported
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported: {settings.SUPPORTED_FORMATS}"
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Max: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_ext == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif file_ext == ".docx":
                # For DOCX, use pypandoc or python-docx
                # This is a placeholder - implement based on needs
                loader = TextLoader(str(file_path))
            else:
                raise ValueError(f"No loader configured for {file_ext}")
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "file_type": file_ext,
                    "file_name": file_path.name
                })
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_directory(self, directory: Path) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of all loaded Document objects
        """
        all_documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in settings.SUPPORTED_FORMATS:
                try:
                    docs = self.load_document(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping {file_path.name}: {e}")
        
        logger.info(
            f"Loaded {len(all_documents)} documents from {directory}"
        )
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunked_docs):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(
            f"Split {len(documents)} documents into {len(chunked_docs)} chunks"
        )
        return chunked_docs
    
    def process_directory(self, directory: Path) -> List[Document]:
        """
        Complete pipeline: load and chunk all documents in directory.
        
        Args:
            directory: Path to document directory
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        logger.info(f"Processing directory: {directory}")
        
        # Load all documents
        documents = self.load_directory(directory)
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        logger.info(
            f"Processing complete: {len(documents)} docs → {len(chunks)} chunks"
        )
        return chunks
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        total_chars = sum(len(doc.page_content) for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars / len(documents) if documents else 0,
            "file_types": file_types,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
