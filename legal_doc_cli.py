#!/usr/bin/env python3
"""
Command-line interface for the Legal Document Analysis System.
This script provides a complete workflow for processing legal case files:
1. Process ZIP files containing case documents
2. Create vector stores for efficient querying
3. Query documents for specific information
4. Generate medical chronologies
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from src.legal_doc_analysis.document_processor import DocumentProcessor
from src.legal_doc_analysis.document_ingestion.zip_processor import process_zip_file
from config.settings import INPUT_DIR, PROCESSED_DIR, VECTOR_STORE_DIR, OPENAI_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('legal_doc_cli.log')
    ]
)
logger = logging.getLogger(__name__)

def process_case(zip_path: Path, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Process a case ZIP file and create a vector store.
    
    Args:
        zip_path: Path to the ZIP file containing case documents
        force_rebuild: Whether to force rebuilding the vector store
        
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    logger.info(f"Processing case file: {zip_path}")
    
    # Extract case name from filename
    case_name = zip_path.stem
    output_dir = PROCESSED_DIR / case_name
    
    # Step 1: Extract ZIP file
    logger.info(f"Extracting ZIP file to {output_dir}")
    zip_result = process_zip_file(zip_path, output_dir)
    
    if not zip_result['success']:
        logger.error(f"Failed to extract ZIP file: {zip_result['error']}")
        return {
            'success': False,
            'error': f"ZIP extraction failed: {zip_result['error']}",
            'processing_time': time.time() - start_time
        }
    
    # Step 2: Process documents and create vector store
    logger.info(f"Processing documents in {output_dir}")
    try:
        processor = DocumentProcessor(
            case_dir=str(output_dir),
            persist_dir=str(VECTOR_STORE_DIR)
        )
        
        # Load documents
        document_count = len(processor.load_documents())
        logger.info(f"Loaded {document_count} documents")
        
        # Create vector store
        vector_store_created = processor.create_vector_store(force_rebuild=force_rebuild)
        logger.info(f"Vector store {'created' if vector_store_created else 'loaded'}")
        
        return {
            'success': True,
            'case_name': case_name,
            'document_count': document_count,
            'vector_store_path': str(processor._get_vectorstore_path()),
            'processing_time': time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error processing case: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time
        }

def query_case(case_name: str, query: str) -> Dict[str, Any]:
    """
    Query a processed case.
    
    Args:
        case_name: Name of the case to query
        query: Question to ask about the case
        
    Returns:
        Dictionary with query results
    """
    case_dir = PROCESSED_DIR / case_name
    
    if not case_dir.exists():
        return {
            'success': False,
            'error': f"Case directory not found: {case_dir}"
        }
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(
            case_dir=str(case_dir),
            persist_dir=str(VECTOR_STORE_DIR)
        )
        
        # Load vector store
        vector_store_loaded = processor.create_vector_store()
        if not vector_store_loaded:
            return {
                'success': False,
                'error': "Failed to load vector store"
            }
        
        # Set up QA chain
        processor.setup_qa_chain()
        
        # Process query
        result = processor.query_documents(query)
        
        # Format sources for output
        sources = []
        if 'source_documents' in result:
            for doc in result['source_documents']:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                sources.append(f"{source} (Page {page})")
        
        return {
            'success': True,
            'answer': result.get('answer', 'No answer found'),
            'sources': sources
        }
        
    except Exception as e:
        logger.error(f"Error querying case: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def generate_chronology(case_name: str) -> Dict[str, Any]:
    """
    Generate a medical chronology for a case.
    
    Args:
        case_name: Name of the case
        
    Returns:
        Dictionary with chronology results
    """
    case_dir = PROCESSED_DIR / case_name
    
    if not case_dir.exists():
        return {
            'success': False,
            'error': f"Case directory not found: {case_dir}"
        }
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(
            case_dir=str(case_dir),
            persist_dir=str(VECTOR_STORE_DIR)
        )
        
        # Load vector store
        vector_store_loaded = processor.create_vector_store()
        if not vector_store_loaded:
            return {
                'success': False,
                'error': "Failed to load vector store"
            }
        
        # Set up QA chain
        processor.setup_qa_chain()
        
        # Generate chronology
        chronology = processor.generate_medical_chronology()
        
        return {
            'success': True,
            'chronology': chronology
        }
        
    except Exception as e:
        logger.error(f"Error generating chronology: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def list_cases() -> List[Dict[str, Any]]:
    """
    List all processed cases.
    
    Returns:
        List of dictionaries with case information
    """
    if not PROCESSED_DIR.exists():
        return []
    
    cases = []
    for case_dir in PROCESSED_DIR.iterdir():
        if case_dir.is_dir():
            # Count files in the case directory
            file_count = len([f for f in case_dir.rglob('*') if f.is_file()])
            
            # Check if vector store exists for this case
            processor = DocumentProcessor(
                case_dir=str(case_dir),
                persist_dir=str(VECTOR_STORE_DIR)
            )
            vector_store_path = processor._get_vectorstore_path()
            vector_store_exists = (vector_store_path.parent / f"{vector_store_path.name}.faiss").exists()
            
            cases.append({
                'name': case_dir.name,
                'path': str(case_dir),
                'file_count': file_count,
                'vector_store_exists': vector_store_exists,
                'last_modified': case_dir.stat().st_mtime
            })
    
    # Sort by last modified time (newest first)
    return sorted(cases, key=lambda x: x['last_modified'], reverse=True)

def interactive_mode():
    """Run the CLI in interactive mode."""
    print("\n=== Legal Document Analysis System ===")
    
    while True:
        print("\nAvailable commands:")
        print("1. Process a new case")
        print("2. List available cases")
        print("3. Query a case")
        print("4. Generate medical chronology")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Process a new case
            zip_files = list(INPUT_DIR.glob('*.zip'))
            
            if not zip_files:
                print(f"\nNo ZIP files found in {INPUT_DIR}")
                print(f"Please add a ZIP file to {INPUT_DIR} and try again")
                continue
            
            print("\nAvailable ZIP files:")
            for i, zip_file in enumerate(zip_files, 1):
                print(f"{i}. {zip_file.name}")
            
            try:
                idx = int(input("\nSelect a ZIP file to process (number): ").strip()) - 1
                if 0 <= idx < len(zip_files):
                    zip_path = zip_files[idx]
                    force_rebuild = input("Force rebuild vector store? (y/N): ").strip().lower() == 'y'
                    
                    print(f"\nProcessing {zip_path.name}...")
                    result = process_case(zip_path, force_rebuild)
                    
                    if result['success']:
                        print(f"\n✅ Successfully processed case: {result['case_name']}")
                        print(f"Documents processed: {result['document_count']}")
                        print(f"Vector store: {result['vector_store_path']}")
                        print(f"Processing time: {result['processing_time']:.2f} seconds")
                    else:
                        print(f"\n❌ Failed to process case: {result['error']}")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            # List available cases
            cases = list_cases()
            
            if not cases:
                print("\nNo processed cases found")
                continue
            
            print("\nAvailable cases (newest first):")
            for i, case in enumerate(cases, 1):
                vs_status = "✅" if case['vector_store_exists'] else "❌"
                print(f"{i}. {case['name']} ({case['file_count']} files) [Vector Store: {vs_status}]")
        
        elif choice == '3':
            # Query a case
            cases = list_cases()
            
            if not cases:
                print("\nNo processed cases found")
                continue
            
            print("\nAvailable cases:")
            for i, case in enumerate(cases, 1):
                print(f"{i}. {case['name']}")
            
            try:
                idx = int(input("\nSelect a case to query (number): ").strip()) - 1
                if 0 <= idx < len(cases):
                    case_name = cases[idx]['name']
                    
                    print(f"\nQuerying case: {case_name}")
                    print("Type 'exit' to return to the main menu")
                    
                    while True:
                        query = input("\nYour question: ").strip()
                        
                        if query.lower() == 'exit':
                            break
                        
                        if not query:
                            continue
                        
                        print("\nProcessing query...")
                        result = query_case(case_name, query)
                        
                        if result['success']:
                            print("\nANSWER:")
                            print("-" * 50)
                            print(result['answer'])
                            print("-" * 50)
                            
                            if result['sources']:
                                print("\nSOURCES:")
                                for i, source in enumerate(result['sources'], 1):
                                    print(f"{i}. {source}")
                        else:
                            print(f"\n❌ Query failed: {result['error']}")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            # Generate medical chronology
            cases = list_cases()
            
            if not cases:
                print("\nNo processed cases found")
                continue
            
            print("\nAvailable cases:")
            for i, case in enumerate(cases, 1):
                print(f"{i}. {case['name']}")
            
            try:
                idx = int(input("\nSelect a case for chronology (number): ").strip()) - 1
                if 0 <= idx < len(cases):
                    case_name = cases[idx]['name']
                    
                    print(f"\nGenerating medical chronology for case: {case_name}")
                    print("This may take a few minutes...")
                    
                    result = generate_chronology(case_name)
                    
                    if result['success']:
                        print("\nMEDICAL CHRONOLOGY:")
                        print("=" * 50)
                        print(result['chronology'])
                        print("=" * 50)
                        
                        # Save chronology to file
                        save_path = PROCESSED_DIR / case_name / "medical_chronology.txt"
                        with open(save_path, 'w') as f:
                            f.write(result['chronology'])
                        print(f"\nChronology saved to: {save_path}")
                    else:
                        print(f"\n❌ Chronology generation failed: {result['error']}")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            # Exit
            print("\nExiting Legal Document Analysis System. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

def main():
    parser = argparse.ArgumentParser(description="Legal Document Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a case ZIP file")
    process_parser.add_argument("zip_path", type=str, help="Path to the ZIP file")
    process_parser.add_argument("--force", action="store_true", help="Force rebuild vector store")
    
    # List command
    subparsers.add_parser("list", help="List available cases")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a case")
    query_parser.add_argument("case_name", type=str, help="Name of the case to query")
    query_parser.add_argument("question", type=str, help="Question to ask about the case")
    
    # Chronology command
    chronology_parser = subparsers.add_parser("chronology", help="Generate medical chronology")
    chronology_parser.add_argument("case_name", type=str, help="Name of the case")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in environment variables")
        print("Please set the OPENAI_API_KEY environment variable and try again")
        return 1
    
    # Run the appropriate command
    if args.command == "process":
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            print(f"Error: ZIP file not found: {zip_path}")
            return 1
        
        result = process_case(zip_path, args.force)
        
        if result['success']:
            print(f"\n✅ Successfully processed case: {result['case_name']}")
            print(f"Documents processed: {result['document_count']}")
            print(f"Vector store: {result['vector_store_path']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
        else:
            print(f"\n❌ Failed to process case: {result['error']}")
    
    elif args.command == "list":
        cases = list_cases()
        
        if not cases:
            print("No processed cases found")
            return 0
        
        print("\nAvailable cases (newest first):")
        for i, case in enumerate(cases, 1):
            vs_status = "✅" if case['vector_store_exists'] else "❌"
            print(f"{i}. {case['name']} ({case['file_count']} files) [Vector Store: {vs_status}]")
    
    elif args.command == "query":
        result = query_case(args.case_name, args.question)
        
        if result['success']:
            print("\nANSWER:")
            print("-" * 50)
            print(result['answer'])
            print("-" * 50)
            
            if result['sources']:
                print("\nSOURCES:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source}")
        else:
            print(f"\n❌ Query failed: {result['error']}")
    
    elif args.command == "chronology":
        print(f"Generating medical chronology for case: {args.case_name}")
        print("This may take a few minutes...")
        
        result = generate_chronology(args.case_name)
        
        if result['success']:
            print("\nMEDICAL CHRONOLOGY:")
            print("=" * 50)
            print(result['chronology'])
            print("=" * 50)
            
            # Save chronology to file
            save_path = PROCESSED_DIR / args.case_name / "medical_chronology.txt"
            with open(save_path, 'w') as f:
                f.write(result['chronology'])
            print(f"\nChronology saved to: {save_path}")
        else:
            print(f"\n❌ Chronology generation failed: {result['error']}")
    
    elif args.command == "interactive" or not args.command:
        interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
