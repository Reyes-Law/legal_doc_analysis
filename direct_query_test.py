#!/usr/bin/env python3
"""
Direct query test for legal document analysis.
This script bypasses the normal document processing pipeline and directly accesses
the existing vector store to test querying functionality.
"""
import os
import sys
import time
import glob
from pathlib import Path
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import required modules
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import project settings
from config.settings import VECTOR_STORE_DIR, OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure OpenAI API key is set
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Constants
CASE_NAME = "austinsmatter"  # Replace with your case name

# Find the vector store files
def find_vector_store():
    """Find the vector store files in the vector store directory."""
    # Look for .faiss files in the vector store directory
    faiss_files = list(VECTOR_STORE_DIR.glob("*.faiss"))
    
    if not faiss_files:
        print(f"Error: No vector store files found in {VECTOR_STORE_DIR}")
        sys.exit(1)
    
    # Use the first .faiss file found
    vector_store_path = Path(faiss_files[0].stem)
    print(f"Found vector store: {vector_store_path.name}")
    return vector_store_path

# Load the vector store
def load_vector_store(vector_store_path):
    """Load the vector store from disk."""
    print("Loading vector store...")
    start_time = time.time()
    
    try:
        # Initialize embeddings with the same model used for creating the vector store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        
        # Load the vector store
        vectorstore = FAISS.load_local(
            folder_path=str(VECTOR_STORE_DIR),
            index_name=vector_store_path.name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        print(f"Vector store contains {len(vectorstore.index_to_docstore_id)} vectors")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vector store: {e}")
        sys.exit(1)

# Set up the QA chain
def setup_qa_chain(vectorstore):
    """Set up the QA chain for querying documents."""
    print("\n=== Setting up QA Chain ===")
    
    # Initialize the language model
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Set up memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    # Create the conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return qa_chain

# Interactive query loop
def interactive_query(qa_chain):
    """Run an interactive query loop."""
    print("\n=== Interactive Query Mode ===")
    print("Type your questions about the legal documents.")
    print("Type 'exit' or 'quit' to end the session.")
    print("\n")
    
    while True:
        query = input("Your question: ")
        
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        
        if not query.strip():
            continue
        
        print("\nProcessing your question...")
        start_time = time.time()
        
        try:
            result = qa_chain({"question": query})
            print(f"Query processed in {time.time() - start_time:.2f} seconds\n")
            
            print("ANSWER:")
            print("-" * 50)
            print(result["answer"])
            print("-" * 50)
            
            print("\nSOURCES:")
            sources = {}
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                if source not in sources:
                    sources[source] = set()
                sources[source].add(str(page))
            
            for i, (source, pages) in enumerate(sources.items(), 1):
                pages_str = f"pages {', '.join(sorted(pages))}" if pages else ""
                print(f"{i}. {source} {pages_str}")
            
            print("\n")
        except Exception as e:
            print(f"Error processing query: {e}")
            print(f"\nError processing query: {e}")

def main():
    # Find vector store
    vector_store_path = find_vector_store()
    
    # Check if vector store exists
    if not (VECTOR_STORE_DIR / f"{vector_store_path.name}.faiss").exists():
        print(f"Error: Vector store not found at {vector_store_path}")
        print(f"Make sure the vector store files exist in {VECTOR_STORE_DIR}")
        sys.exit(1)
    
    # Load vector store
    vectorstore = load_vector_store(vector_store_path)
    
    # Set up QA chain
    qa_chain = setup_qa_chain(vectorstore)
    
    # Run interactive query loop
    interactive_query(qa_chain)

if __name__ == "__main__":
    main()
