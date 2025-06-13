from src.legal_doc_analysis.document_processor import DocumentProcessor
import os

class FastDocumentProcessor(DocumentProcessor):
    def create_vector_store(self, force_rebuild=False):
        """Override to prevent rebuilding the vector store"""
        vectorstore_path = self._get_vectorstore_path()
        if os.path.exists(vectorstore_path):
            print("Using existing vector store")
            return True
        return False

print("Initializing with existing vector store...")
processor = FastDocumentProcessor('data/processed/MAT-23012393855 copy', 'data/vectorstores')

# This should now use the existing vector store
print("\nTesting query...")
result = processor.query_documents("What is the policy number?")
print("\nANSWER:", result.get("answer", "No answer found"))
