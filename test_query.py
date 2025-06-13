from src.legal_doc_analysis.document_processor import DocumentProcessor
import time

print("Initializing...")
start = time.time()
processor = DocumentProcessor('data/processed/austinsmatter', 'data/vectorstores')
print(f"Initialized in {time.time() - start:.1f} seconds")

print("\nTesting simple query...")
result = processor.query_documents("What is the policy number?")
print("\nANSWER:", result.get("answer", "No answer found"))
