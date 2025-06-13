#!/usr/bin/env python3
"""Quick test script for adaptive RAG system."""
import os
import sys
from pathlib import Path
from src.legal_doc_analysis.document_processor import DocumentProcessor

def test_query(processor, query):
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print("-" * 40)
    
    result = processor.query_documents(query)
    
    print(f"QUERY TYPE: {result.get('query_type', 'unknown')}")
    print(f"SOURCES ({len(result.get('sources', []))}):")
    for src in result.get('sources', [])[:5]:  # Show first 5 sources
        print(f"- {src}")
    if len(result.get('sources', [])) > 5:
        print(f"... and {len(result['sources']) - 5} more")
    
    print("\nRESPONSE:")
    print(result.get('answer', 'No answer generated'))
    
    if 'error' in result:
        print(f"\nERROR: {result['error']}")

def main():
    # Initialize with the actual case directory
    case_dir = "data/processed/MAT-23012393855 copy"  # Using the actual case directory
    vectorstore_dir = "data/vectorstores"
    
    print(f"Initializing with case directory: {case_dir}")
    print(f"Vectorstore directory: {vectorstore_dir}")
    
    # Check if case directory exists
    if not os.path.exists(case_dir):
        print(f"Error: Case directory not found at {case_dir}")
        print("Please ensure the directory exists and contains your case documents.")
        return
    
    processor = DocumentProcessor(case_dir, vectorstore_dir)
    
    # Test queries
    test_queries = [
        # Factual
        "What is the patient's date of birth?",
        "List all prescribed medications.",
        
        # Synthesis
        "Generate a medical chronology",
        "Summarize all test results",
        
        # Mixed
        "What medications were prescribed and show me a treatment history"
    ]
    
    for query in test_queries:
        test_query(processor, query)
        input("\nPress Enter to continue to next test...")

if __name__ == "__main__":
    main()
