import os
from src.legal_doc_analysis.document_processor import DocumentProcessor
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

class FastDocumentProcessor(DocumentProcessor):
    def create_vector_store(self, force_rebuild=False):
        vectorstore_path = self._get_vectorstore_path()
        if os.path.exists(vectorstore_path) and not force_rebuild:
            print("Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
                allow_dangerous_deserialization=True
            )
            return True
        return False

print("Initializing with existing vector store...")
processor = FastDocumentProcessor('data/processed/MAT-23012393855 copy', 'data/vectorstores')

# This should now use the existing vector store
print("\nTesting query...")
result = processor.query_documents("What is the policy number?")
print("\nANSWER:", result.get("answer", "No answer found"))
