
"""
Document ingestion script for RAG system.
Processes documents from a directory and builds vector store.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_engine import RAGEngine
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG vector store"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(settings.RAW_DATA_DIR),
        help="Source directory containing documents"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.VECTOR_STORE_PATH,
        help="Output path for vector store"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE,
        help="Text chunk size"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.CHUNK_OVERLAP,
        help="Chunk overlap size"
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    source_dir = Path(args.source)
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Starting Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Source Directory: {source_dir}")
    logger.info(f"Output Path: {args.output}")
    logger.info(f"Chunk Size: {args.chunk_size}")
    logger.info(f"Chunk Overlap: {args.chunk_overlap}")
    logger.info("=" * 60)
    
    try:
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine()
        
        # Ingest documents
        logger.info(f"Processing documents from {source_dir}...")
        result = rag_engine.ingest_documents(source_dir)
        
        if result["status"] == "error":
            logger.error(f"Ingestion failed: {result.get('message')}")
            sys.exit(1)
        
        # Save vector store
        logger.info(f"Saving vector store to {args.output}...")
        rag_engine.save_vector_store(Path(args.output))
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Ingestion Complete!")
        logger.info("=" * 60)
        logger.info(f"Documents Processed: {result['documents_processed']}")
        logger.info(f"Vector Store Stats: {result['vector_store_stats']}")
        logger.info("=" * 60)
        
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: python app/main.py")
        logger.info("2. Or query directly: python scripts/query.py")
        
    except Exception as e:
        logger.error(f"Ingestion failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
