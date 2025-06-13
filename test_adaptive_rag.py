#!/usr/bin/env python3
"""Test script for adaptive RAG system."""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.legal_doc_analysis.document_processor import DocumentProcessor
from config.settings import PROCESSED_DIR

def run_test_case(processor: DocumentProcessor, query: str, expected_type: str) -> Dict[str, Any]:
    """Run a single test case and return results."""
    print(f"\n{'='*80}")
    print(f"TESTING: {query}")
    print(f"Expected type: {expected_type}")
    print("-" * 40)
    
    start_time = time.time()
    try:
        result = processor.query_documents(query)
        duration = time.time() - start_time
        
        print(f"RESULT (took {duration:.2f}s):")
        print(f"Query type: {result.get('query_type', 'unknown')}")
        print(f"Sources: {len(result.get('sources', []))} unique sources")
        print("\nANSWER:")
        print(result.get('answer', 'No answer generated'))
        
        if 'error' in result:
            print(f"\nERROR: {result['error']}")
            
        return {
            'success': 'error' not in result,
            'duration': duration,
            'sources_count': len(result.get('sources', [])),
            'query_type': result.get('query_type')
        }
        
    except Exception as e:
        print(f"TEST FAILED: {str(e)}")
        return {'success': False, 'error': str(e)}

def main():
    """Run test cases for adaptive RAG system."""
    # Initialize processor with test case directory
    case_dir = PROCESSED_DIR / "MAT-23012393855 copy"  # Update with your test case
    processor = DocumentProcessor(str(case_dir), "data/vectorstores")
    
    # Test cases
    test_cases = [
        # Factual queries
        ("What is the policy number?", "factual"),
        ("When was the last doctor's visit?", "factual"),
        ("What medications were prescribed?", "factual"),
        
        # Synthesis queries
        ("Generate a medical chronology", "synthesis"),
        ("List all medical treatments and procedures", "synthesis"),
        ("Summarize all test results and findings", "synthesis"),
        
        # Mixed/edge cases
        ("What is the policy number and also show me a treatment history", "mixed"),
        ("", "error"),  # Empty query
    ]
    
    # Run tests
    results = []
    for query, expected_type in test_cases:
        result = run_test_case(processor, query, expected_type)
        results.append({
            'query': query,
            'expected_type': expected_type,
            **result
        })
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r.get('success'))
    print(f"Tests completed: {len(results)}")
    print(f"Tests passed: {success_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    for i, r in enumerate(results, 1):
        status = "PASS" if r.get('success') else "FAIL"
        print(f"{i}. [{status}] {r['query'][:60]}...")
        print(f"   Type: {r.get('query_type')} (expected: {r.get('expected_type')})")
        print(f"   Time: {r.get('duration', 0):.2f}s, Sources: {r.get('sources_count', 0)}")
        if 'error' in r:
            print(f"   Error: {r['error']}")

if __name__ == "__main__":
    main()
