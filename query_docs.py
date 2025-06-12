#!/usr/bin/env python3
"""Command-line interface for querying legal documents."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.legal_doc_analysis.document_processor import DocumentProcessor
from config.settings import PROCESSED_DIR, OPENAI_API_KEY

def ensure_vectorstore_dir():
    """Ensure the vectorstore directory exists."""
    vectorstore_dir = Path("data/vectorstores")
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return vectorstore_dir

def list_available_cases():
    print("\nAvailable processed cases:")
    if not case_dirs:
        print("\nNo processed cases found in the data/processed directory.")
        return None
        
    print("\nAvailable processed cases (newest first):")
    for i, case_dir in enumerate(case_dirs, 1):
        # Count number of files in the case directory
        num_files = len([f for f in case_dir.rglob('*') if f.is_file()])
        print(f"{i}. {case_dir.name} ({num_files} files)")
    
    while True:
        selection = input("\nSelect a case number (or press Enter to exit): ").strip()
        if not selection:
            return None
            
        try:
            case_idx = int(selection) - 1
            if 0 <= case_idx < len(case_dirs):
                return case_dirs[case_idx]
            print(f"Please enter a number between 1 and {len(case_dirs)}.")
        except ValueError:
            print("Please enter a valid number.")

def get_available_cases() -> List[Path]:
    """Get a list of available case directories, sorted by modification time (newest first)."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted(
        [d for d in PROCESSED_DIR.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

def select_case_interactively() -> Optional[Path]:
    """Show interactive case selection prompt and return selected case path."""
    case_dirs = get_available_cases()
    
    if not case_dirs:
        print("\nNo processed cases found in the data/processed directory.")
        return None
        
    print("\nAvailable processed cases (newest first):")
    for i, case_dir in enumerate(case_dirs, 1):
        # Count number of files in the case directory
        num_files = len([f for f in case_dir.rglob('*') if f.is_file()])
        print(f"{i}. {case_dir.name} ({num_files} files)")
    
    while True:
        selection = input("\nSelect a case number (or press Enter to exit): ").strip()
        if not selection:
            return None
            
        try:
            case_idx = int(selection) - 1
            if 0 <= case_idx < len(case_dirs):
                return case_dirs[case_idx]
            print(f"Please enter a number between 1 and {len(case_dirs)}.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    parser = argparse.ArgumentParser(description="Query legal documents using natural language.")
    parser.add_argument('case_dir', type=str, nargs='?',
                      help='Path to the case directory (default: interactive selection)')
    parser.add_argument('--list-cases', action='store_true',
                      help='List available processed cases and exit')
    parser.add_argument('--rebuild', action='store_true',
                      help='Force rebuild the vector store')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('query_docs.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Handle case directory selection
        if args.list_cases:
            case_dirs = get_available_cases()
            if not case_dirs:
                print("No processed cases found in the data/processed directory.")
                return
                
            print("\nAvailable processed cases (newest first):")
            for i, case_dir in enumerate(case_dirs, 1):
                num_files = len([f for f in case_dir.rglob('*') if f.is_file()])
                print(f"{i}. {case_dir.name} ({num_files} files)")
            return
        
        # Get the case directory
        if args.case_dir:
            case_dir = Path(args.case_dir)
            if not case_dir.exists():
                print(f"Error: Directory '{case_dir}' not found.")
                return
        else:
            case_dir = select_case_interactively()
            if not case_dir:
                return
                
        logger.info(f"Selected case directory: {case_dir}")
        
        # Ensure vectorstore directory exists
        vectorstore_dir = Path("data/vectorstores")
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize document processor
        processor = DocumentProcessor(
            case_dir=str(case_dir),
            persist_dir=str(vectorstore_dir)
        )
        
        # Load and process documents
        print("\nLoading and processing documents (this may take a few minutes for the first time)...")
        try:
            # Load documents with progress feedback
            documents = processor.load_documents()
            if not documents:
                print("\nNo valid documents found to process.")
                return
                
            print(f"\nFound {len(documents)} document pages to process.")
            
            # Create or load vector store
            print("\nCreating/loading vector store...")
            was_cached = processor.create_vector_store(force_rebuild=args.rebuild)
            if was_cached:
                print("Loaded existing vector store from cache.")
            else:
                print("Created new vector store.")
                
            # Set up QA chain
            print("\nSetting up QA chain...")
            processor.setup_qa_chain(force_rebuild=args.rebuild)
            
            # Interactive query loop
            print("\nDocument processing complete!")
            print("You can now ask questions about the case or type 'chronology' to generate a medical chronology.")
            print("Type 'exit' or 'quit' to end the session.\n")
            
            while True:
                try:
                    query = input("\nYour question: ").strip()
                    
                    if query.lower() in ['exit', 'quit']:
                        print("Goodbye!")
                        break
                        
                    if not query:
                        continue
                        
                    if query.lower() == 'chronology':
                        print("\nGenerating medical chronology...")
                        chronology = processor.generate_medical_chronology()
                        print("\nMEDICAL CHRONOLOGY:")
                        print("-" * 50)
                        print(chronology)
                        print("-" * 50)
                        continue
                        
                    # Process the query
                    print("\nProcessing your question...")
                    result = processor.query_documents(query)
                    
                    # Display results
                    if 'answer' in result:
                        print("\nANSWER:")
                        print("-" * 50)
                        print(result['answer'])
                        print("-" * 50)
                        
                        # Show source documents
                        if 'source_documents' in result and result['source_documents']:
                            print("\nSOURCES:")
                            sources = {}
                            for i, doc in enumerate(result['source_documents'], 1):
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'N/A')
                                if source not in sources:
                                    sources[source] = set()
                                sources[source].add(page)
                            
                            for i, (source, pages) in enumerate(sources.items(), 1):
                                pages_str = f"pages {', '.join(sorted(pages))}" if pages else ""
                                print(f"{i}. {source} {pages_str}")
                    
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {e}", exc_info=args.debug)
                    print(f"\nAn error occurred: {str(e)}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
        
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}", exc_info=args.debug)
            print(f"\nError initializing document processor: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    print(f"\nLoading documents from: {case_dir}")
    
    try:
        vectorstore_dir = ensure_vectorstore_dir()
        processor = DocumentProcessor(
            case_dir=str(case_dir),
            persist_dir=str(vectorstore_dir)
        )
        logger = logging.getLogger(__name__)
        logger.info("Loading and processing documents (this may take a few minutes for the first time)...")
        processor.load_documents()
        processor.create_vector_store()
        processor.setup_qa_chain(force_rebuild=args.rebuild)
        
        print("\nDocument processing complete!")
        print("You can now ask questions about the case or type 'chronology' to generate a medical chronology.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        # Interactive query loop
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ('exit', 'quit'):
                    break
                    
                if not query:
                    continue
                    
                if query.lower() == 'chronology':
                    print("\nGenerating medical chronology...")
                    chronology = processor.generate_medical_chronology()
                    print("\n=== MEDICAL CHRONOLOGY ===\n")
                    print(chronology)
                    continue
                
                print("\nSearching documents...")
                result = processor.query_documents(query)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    continue
                
                print("\n=== ANSWER ===")
                print(result["answer"])
                
                if result["sources"]:
                    print("\nSources:")
                    for source in result["sources"]:
                        print(f"- {source}")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to end the session.")
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except Exception as e:
        print(f"Error initializing document processor: {e}")
        if "OPENAI_API_KEY" in str(e):
            print("\nPlease make sure you have set the OPENAI_API_KEY in your .env file.")

if __name__ == "__main__":
    main()
