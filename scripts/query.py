"""
Command-line interface for querying the RAG system.
Supports both single queries and interactive mode.
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


def print_response(result: dict):
    """Pretty print query response."""
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("\n" + "=" * 80)
    print(f"Retrieved Documents: {result['retrieved_docs']}")
    
    if result.get("sources"):
        print("\nSOURCES:")
        print("-" * 80)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['file_name']}")
            print(f"    Relevance Score: {source['score']:.3f}")
            print(f"    Preview: {source['preview']}")
    
    if result.get("usage"):
        print("\n" + "-" * 80)
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Tokens Used: {result['usage']}")
    
    print("=" * 80 + "\n")


def interactive_mode(rag_engine: RAGEngine, top_k: int):
    """Run interactive query session."""
    print("\n" + "=" * 80)
    print("Interactive RAG Query Mode")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end session")
    print("Type 'stats' to see system statistics")
    print("=" * 80 + "\n")
    
    while True:
        try:
            question = input("\nYour Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
            
            if question.lower() == "stats":
                stats = rag_engine.get_stats()
                print("\nSystem Statistics:")
                print(f"  Total Documents: {stats['vector_store']['total_documents']}")
                print(f"  Model: {stats['configuration']['model']}")
                print(f"  Embedding Model: {stats['configuration']['embedding_model']}")
                continue
            
            print("\nProcessing your question...")
            result = rag_engine.query(
                question=question,
                top_k=top_k,
                include_sources=True
            )
            
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Query error: {e}", exc_info=True)


def single_query(rag_engine: RAGEngine, question: str, top_k: int):
    """Execute single query."""
    logger.info(f"Processing query: {question}")
    
    result = rag_engine.query(
        question=question,
        top_k=top_k,
        include_sources=True
    )
    
    print_response(result)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Query the RAG system from command line"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask (for single query mode)"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=settings.TOP_K_RETRIEVAL,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=settings.VECTOR_STORE_PATH,
        help="Path to vector store"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.question:
        parser.error("Either --question or --interactive must be specified")
    
    # Initialize RAG engine
    try:
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine()
        
        # Load vector store
        vector_store_path = Path(args.vector_store)
        if not vector_store_path.exists():
            logger.error(f"Vector store not found at {vector_store_path}")
            logger.error("Please run 'python scripts/ingest.py' first")
            sys.exit(1)
        
        logger.info(f"Loading vector store from {vector_store_path}...")
        rag_engine.load_vector_store(vector_store_path)
        
        stats = rag_engine.get_stats()
        logger.info(f"Loaded {stats['vector_store']['total_documents']} documents")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        sys.exit(1)
    
    # Run query mode
    try:
        if args.interactive:
            interactive_mode(rag_engine, args.top_k)
        else:
            single_query(rag_engine, args.question, args.top_k)
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
