
"""
FAISS vector store for efficient similarity search.
Handles embedding storage, indexing, and retrieval.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from langchain.schema import Document

from app.services.bedrock_client import BedrockClient
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        """
        Initialize vector store.
        
        Args:
            bedrock_client: BedrockClient instance for embeddings
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.dimension = settings.EMBEDDING_DIMENSION
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.doc_embeddings: List[np.ndarray] = []
        
        logger.info(f"Initialized VectorStore with dimension={self.dimension}")
    
    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index for similarity search.
        
        Returns:
            FAISS index instance
        """
        # Use L2 distance (Euclidean) for similarity
        # For production, consider IndexIVFFlat for larger datasets
        index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created FAISS index (L2 distance, dim={self.dimension})")
        return index
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector store with embeddings.
        
        Args:
            documents: List of Document objects to index
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.bedrock_client.generate_embeddings(batch)
            all_embeddings.extend(embeddings)
            
            logger.info(
                f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
            )
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create or update index
        if self.index is None:
            self.index = self._create_index()
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.doc_embeddings.extend(all_embeddings)
        
        logger.info(
            f"Successfully added {len(documents)} documents. "
            f"Total documents: {len(self.documents)}"
        )
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        score_threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or settings.TOP_K_RETRIEVAL
        score_threshold = score_threshold or settings.SIMILARITY_THRESHOLD
        
        if self.index is None or len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.bedrock_client.generate_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_vector, k)
        
        # Convert distances to similarity scores (inverse of L2 distance)
        # Lower distance = higher similarity
        similarities = 1 / (1 + distances[0])
        
        # Filter by threshold and prepare results
        results = []
        for idx, score in zip(indices[0], similarities):
            if idx < len(self.documents) and score >= score_threshold:
                doc = self.documents[idx]
                results.append((doc, float(score)))
        
        logger.info(
            f"Retrieved {len(results)} documents for query (k={k}, "
            f"threshold={score_threshold})"
        )
        
        return results
    
    def save(self, path: Path) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save vector store
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(path / "embeddings.pkl", "wb") as f:
            pickle.dump(self.doc_embeddings, f)
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path containing saved vector store
        """
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        # Load FAISS index
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        
        # Load embeddings
        with open(path / "embeddings.pkl", "rb") as f:
            self.doc_embeddings = pickle.load(f)
        
        logger.info(
            f"Loaded vector store from {path}: "
            f"{len(self.documents)} documents"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None
        }
    
    def delete_all(self) -> None:
        """Clear all documents and reset index."""
        self.index = None
        self.documents = []
        self.doc_embeddings = []
        logger.info("Cleared vector store")
