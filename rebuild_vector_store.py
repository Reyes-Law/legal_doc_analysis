#!/usr/bin/env python3
"""
Script to rebuild the vector store for a specific case.
This forces a complete rebuild of the vector store from the processed documents.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the DocumentProcessor
from src.legal_doc_analysis.document_processor import DocumentProcessor
from config.settings import PROCESSED_DIR

def rebuild_vector_store(case_name):
    """Rebuild the vector store for a specific case."""
    case_dir = PROCESSED_DIR / case_name
    
    if not case_dir.exists():
        logger.error(f"Case directory not found: {case_dir}")
        return False
    
    logger.info(f"Rebuilding vector store for case: {case_name}")
    logger.info(f"Case directory: {case_dir}")
    
    try:
        # Initialize the document processor with the vector store directory
        from config.settings import VECTOR_STORE_DIR
        processor = DocumentProcessor(
            case_dir=case_dir,
            persist_dir=VECTOR_STORE_DIR
        )
        
        # Load documents
        num_docs = processor.load_documents()
        logger.info(f"Loaded {num_docs} documents")
        
        # Force rebuild of vector store
        logger.info("Forcing rebuild of vector store...")
        success = processor.create_vector_store(force_rebuild=True)
        
        if success:
            logger.info("✅ Successfully rebuilt vector store")
            return True
        else:
            logger.error("❌ Failed to rebuild vector store")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error rebuilding vector store: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rebuild_vector_store.py <case_name>")
        print("Available cases:")
        for case_dir in PROCESSED_DIR.iterdir():
            if case_dir.is_dir():
                print(f"  - {case_dir.name}")
        sys.exit(1)
        
    case_name = sys.argv[1]
    rebuild_vector_store(case_name)
