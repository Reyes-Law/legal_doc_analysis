#!/usr/bin/env python3
"""
Test script for medical chronology generation.
This script tests the medical chronology feature directly.
"""
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.legal_doc_analysis.document_processor import DocumentProcessor
from config.settings import PROCESSED_DIR, VECTOR_STORE_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_chronology_generation(case_name: str):
    """Test medical chronology generation for a specific case."""
    print(f"\n{'='*60}")
    print(f"Testing Medical Chronology Generation for: {case_name}")
    print(f"{'='*60}")
    
    case_dir = PROCESSED_DIR / case_name
    
    if not case_dir.exists():
        print(f"❌ Case directory not found: {case_dir}")
        return False
    
    print(f"✅ Case directory found: {case_dir}")
    
    try:
        # Initialize document processor
        print("🔄 Initializing document processor...")
        processor = DocumentProcessor(
            case_dir=str(case_dir),
            persist_dir=str(VECTOR_STORE_DIR)
        )
        
        # Load vector store
        print("🔄 Loading vector store...")
        vector_store_loaded = processor.create_vector_store()
        if not vector_store_loaded:
            print("❌ Failed to load vector store")
            return False
        
        print("✅ Vector store loaded successfully")
        
        # Set up QA chain
        print("🔄 Setting up QA chain...")
        processor.setup_qa_chain()
        print("✅ QA chain set up successfully")
        
        # Test document retrieval
        print("🔄 Testing medical document retrieval...")
        medical_docs = processor._retrieve_medical_documents()
        print(f"📊 Found {len(medical_docs)} medical documents")
        
        if medical_docs:
            print("📄 Sample medical documents:")
            for i, doc in enumerate(medical_docs[:3]):
                source = doc.metadata.get('source', 'unknown')
                content_preview = doc.page_content[:200].replace('\n', ' ')
                print(f"   {i+1}. {source}: {content_preview}...")
        
        # Generate chronology
        print("🔄 Generating medical chronology...")
        chronology = processor.generate_medical_chronology()
        
        print(f"\n{'='*60}")
        print("MEDICAL CHRONOLOGY RESULT:")
        print(f"{'='*60}")
        print(f"Length: {len(chronology) if chronology else 0} characters")
        print(f"Content:\n{chronology}")
        print(f"{'='*60}")
        
        # Analyze the result
        if not chronology:
            print("❌ No chronology generated")
            return False
        elif "Error" in chronology or "error" in chronology:
            print("⚠️  Error in chronology generation")
            return False
        elif len(chronology.strip()) < 50:
            print("⚠️  Chronology seems too short")
            return False
        elif "I don't know" in chronology:
            print("⚠️  Chronology contains 'I don't know'")
            return False
        else:
            print("✅ Chronology generated successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        logger.error(f"Error testing chronology: {e}", exc_info=True)
        return False

def main():
    """Main function to test chronology generation."""
    # List available cases
    processed_cases = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    
    print("Available cases:")
    for i, case in enumerate(processed_cases, 1):
        print(f"  {i}. {case}")
    
    # Test each case
    results = {}
    for case in processed_cases:
        results[case] = test_chronology_generation(case)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS:")
    print(f"{'='*60}")
    for case, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{case}: {status}")

if __name__ == "__main__":
    main()
