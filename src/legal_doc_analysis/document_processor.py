"""Document processing and querying module for legal document analysis."""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import json
import hashlib
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
import numpy as np

# Constants for document processing
MAX_CHUNK_SIZE = 500    # Reduced chunk size to handle large documents
CHUNK_OVERLAP = 100     # Reduced overlap to better manage token limits
BATCH_SIZE = 3          # Smaller batch size for large documents
MAX_DOCUMENT_SIZE_MB = 10  # Maximum document size in MB to process

from config.settings import PROCESSED_DIR, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, processing, and querying."""
    
    def __init__(self, case_dir: str, persist_dir: str = None):
        """Initialize with a case directory containing processed documents.
        
        Args:
            case_dir: Path to the directory containing case documents
            persist_dir: Directory to store/load the vector store (optional)
        """
        self.case_dir = Path(case_dir)
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create persist directory if it doesn't exist
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_vectorstore_path(self) -> Path:
        """Get the path to save/load the vector store."""
        if not self.persist_dir:
            return None
            
        # Use a consistent identifier based on the case directory name
        # This assumes the case directory name is unique and doesn't change
        case_name = self.case_dir.name.strip()
        case_hash = hashlib.md5(case_name.encode()).hexdigest()
        
        return self.persist_dir / f"faiss_index_{case_hash}"
    
    def _load_vectorstore(self) -> bool:
        """Load the vector store from disk if it exists."""
        if not self.persist_dir:
            return False
            
        vectorstore_path = self._get_vectorstore_path()
        if not vectorstore_path.exists():
            return False
            
        try:
            logger.info(f"Loading vector store from {vectorstore_path}")
            self.vectorstore = FAISS.load_local(
                folder_path=str(vectorstore_path.parent),
                index_name=vectorstore_path.name,
                embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            )
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def _save_vectorstore(self):
        """Save the vector store to disk."""
        if not self.persist_dir or not self.vectorstore:
            return
            
        vectorstore_path = self._get_vectorstore_path()
        try:
            logger.info(f"Saving vector store to {vectorstore_path}")
            self.vectorstore.save_local(
                folder_path=str(vectorstore_path.parent),
                index_name=vectorstore_path.name
            )
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def _get_document_fingerprint(self) -> str:
        """Generate a fingerprint of the document set for change detection."""
        file_info = []
        for file_path in self.case_dir.rglob('*'):
            if file_path.is_file():
                file_info.append(f"{file_path}:{file_path.stat().st_mtime}")
        return hashlib.md5(''.join(sorted(file_info)).encode()).hexdigest()
    
    def _is_document_too_large(self, file_path: Path) -> bool:
        """Check if a document is too large to process."""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
            if file_size_mb > MAX_DOCUMENT_SIZE_MB:
                logger.warning(f"Document too large: {file_path.name} ({file_size_mb:.2f}MB > {MAX_DOCUMENT_SIZE_MB}MB)")
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking file size for {file_path.name}: {str(e)}")
            return True  # Skip if we can't check size
        
    def _process_document(self, file_path: Path) -> List[Document]:
        """Process a single document and return its pages/chunks."""
        try:
            logger.debug(f"Processing {file_path.relative_to(self.case_dir)}")
            
            # Select appropriate loader based on file extension
            try:
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    loader = Docx2txtLoader(str(file_path))
                elif file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path))
                else:
                    loader = UnstructuredFileLoader(str(file_path))
                
                # Load document with timeout
                docs = loader.load()
                
                if not docs:
                    logger.warning(f"No content extracted from {file_path.name}")
                    return []
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source': str(file_path.relative_to(self.case_dir)),
                        'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        'file_type': file_path.suffix.lower(),
                        'page': doc.metadata.get('page', 0) + 1  # 1-based page numbers
                    })
                
                logger.debug(f"Extracted {len(docs)} pages from {file_path.name}")
                return docs
                
            except Exception as load_error:
                logger.error(f"Error loading {file_path.name}: {str(load_error)}")
                return []
                
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path.name}: {str(e)}", exc_info=True)
            return []
    
    def load_documents(self) -> List[Document]:
        """Load and process all documents in the case directory."""
        if not self.case_dir.exists():
            raise ValueError(f"Case directory not found: {self.case_dir}")
            
        logger.info(f"Loading documents from: {self.case_dir}")
        documents = []
        
        # Get all document files recursively with progress tracking
        doc_files = []
        exts = ["*.pdf", "*.docx", "*.doc", "*.txt"]
        
        logger.info("Scanning for document files...")
        for ext in exts:
            try:
                files = list(self.case_dir.rglob(ext))
                doc_files.extend(files)
                logger.debug(f"Found {len(files)} {ext} files")
            except Exception as e:
                logger.warning(f"Error scanning for {ext} files: {e}")
        
        # Log file count without listing all files
        logger.info(f"Found {len(doc_files)} total document files")
        
        if not doc_files:
            logger.warning(f"No documents found in {self.case_dir}")
            return []
            
        # Process documents with progress tracking
        total_files = len(doc_files)
        processed_files = 0
        processed_pages = 0
        
        for file_path in sorted(doc_files):
            processed_files += 1
            try:
                if self._is_document_too_large(file_path):
                    logger.warning(f"Skipping large file: {file_path.relative_to(self.case_dir)}")
                    continue
                    
                # Process the document
                docs = self._process_document(file_path)
                if docs:
                    documents.extend(docs)
                    processed_pages += len(docs)
                    
                # Log progress every 10 files or 100 pages
                if processed_files % 10 == 0 or processed_pages >= 100:
                    logger.info(
                        f"Progress: {processed_files}/{total_files} files, "
                        f"{processed_pages} pages processed"
                    )
                    processed_pages = 0  # Reset page counter
                    
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
                
        self.documents = documents
        logger.info(f"Total pages loaded: {len(documents)}")
        return documents
    
    def split_documents(self, chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
        """Split documents into chunks for processing with better handling of large documents."""
        if not self.documents:
            self.load_documents()
        
        # Use defaults if not specified
        chunk_size = chunk_size or MAX_CHUNK_SIZE
        chunk_overlap = chunk_overlap or CHUNK_OVERLAP
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=['\n\n', '\n', '. ', ' ', '']  # Better splitting for legal docs
        )
        
        split_docs = []
        for doc in self.documents:
            try:
                # Split each document individually to better handle large ones
                chunks = text_splitter.split_documents([doc])
                split_docs.extend(chunks)
            except Exception as e:
                logger.warning(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {e}")
                # If splitting fails, add the document as-is
                split_docs.append(doc)
        
        logger.info(f"Split {len(self.documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
    def create_vector_store(self, force_rebuild: bool = False) -> bool:
        """Create or load a vector store from documents.
        
        Args:
            force_rebuild: If True, force rebuild the vector store even if it exists
            
        Returns:
            bool: True if vector store was loaded from cache, False if rebuilt
        """
        # Try to load from disk first if not forcing rebuild
        if not force_rebuild and self._load_vectorstore():
            logger.info("Successfully loaded vector store from disk")
            return True
            
        # Need to rebuild the vector store
        documents = self.split_documents(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not documents:
            logger.warning("No documents to process")
            return False
            
        logger.info(f"Creating vector store from {len(documents)} document chunks...")
        
        # Initialize batch size for this operation
        current_batch_size = BATCH_SIZE
        
        try:
            # Initialize embeddings with rate limiting
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-small",
                request_timeout=60,  # Increased timeout
                max_retries=3,       # Retry on failure
                show_progress_bar=True
            )
            
            # Process documents in smaller batches
            i = 0
            while i < len(documents):
                batch = documents[i:i + current_batch_size]
                current_batch_num = (i // current_batch_size) + 1
                total_batches = (len(documents) + current_batch_size - 1) // current_batch_size
                
                logger.info(f"Processing batch {current_batch_num}/{total_batches} "
                           f"(chunks {i+1}-{min(i+current_batch_size, len(documents))} of {len(documents)})")
                
                try:
                    if self.vectorstore is None:
                        self.vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        self.vectorstore.add_documents(batch)
                        
                    # Save progress after each batch
                    self._save_vectorstore()
                    i += current_batch_size  # Only increment on success
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    logger.error(f"Error processing batch {current_batch_num}: {error_msg}")
                    
                    # Try with smaller batch size if we hit rate limits
                    if any(term in error_msg for term in ["rate limit", "too many", "overloaded"]):
                        current_batch_size = max(1, current_batch_size // 2)
                        logger.warning(f"Rate limit hit, reducing batch size to {current_batch_size}")
                        if current_batch_size == 0:
                            logger.error("Batch size reduced to 0, cannot proceed")
                            raise
                        continue
                        
                    # For other errors, re-raise
                    raise
            
            logger.info("Vector store created/updated successfully")
            return False
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def setup_qa_chain(self, force_rebuild: bool = False):
        """Set up the question-answering chain.
        
        Args:
            force_rebuild: If True, force rebuild the vector store even if it exists
        """
        if self.vectorstore is None:
            was_cached = self.create_vector_store(force_rebuild=force_rebuild)
            if not was_cached:
                logger.info("Vector store was rebuilt, saving to disk...")
                self._save_vectorstore()
        
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for better diversity
            search_kwargs={
                "k": 5,           # Number of documents to return
                "fetch_k": 10,    # Number of documents to fetch before filtering
                "lambda_mult": 0.5  # Diversity parameter (0 = max diversity, 1 = no diversity)
            }
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            max_tokens_limit=4000  # Limit context window size
        )
        return self.qa_chain
    
    def query_documents(self, query: str) -> Dict[str, Any]:
        """Query the document store."""
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        try:
            result = self.qa_chain({"question": query, "chat_history": []})
            return {
                "answer": result["answer"],
                "sources": list(set(doc.metadata["source"] for doc in result["source_documents"]))
            }
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return {"error": str(e)}
    
    def generate_medical_chronology(self) -> str:
        """Generate a comprehensive medical chronology from the documents.
        
        Returns:
            str: Formatted medical chronology with dates, providers, treatments, and findings
        """
        if not self.vectorstore:
            self.setup_qa_chain()
            
        # Step 1: Extract all medical events with detailed context
        extraction_prompt = """
        Extract the following information for each medical event in the documents:
        
        1. Date of service (convert to YYYY-MM-DD format if possible)
        2. Patient's full name
        3. Healthcare provider/facility name
        4. Type of visit/treatment/procedure
        5. Diagnoses or conditions addressed
        6. Treatments provided or prescribed
        7. Test results or findings
        8. Recommendations or follow-up plans
        9. Any notable symptoms or complaints
        
        For each event, provide as much detail as possible from the source documents.
        If a date isn't available, note it as [DATE UNKNOWN].
        """
        
        logger.info("Extracting medical events from documents...")
        events = self.vectorstore.similarity_search(extraction_prompt, k=20)  # Get top 20 most relevant chunks
        
        # Format the events into a structured prompt
        events_text = "\n---\n".join([
            f"Document {i+1}:\n{doc.page_content[:1000]}..." 
            for i, doc in enumerate(events)
        ])
        
        # Step 2: Generate the formatted chronology
        chronology_prompt = f"""
        Based on the following medical events extracted from the documents, 
        create a detailed medical chronology in strict chronological order.
        
        FORMAT REQUIREMENTS:
        - Each entry MUST start with the date in YYYY-MM-DD format
        - If no date is available, use [DATE UNKNOWN]
        - Include all relevant medical details
        - Be concise but thorough
        - Use medical terminology accurately
        - Group related events on the same date
        - Highlight any significant findings or changes in condition
        
        MEDICAL EVENTS:
        {events_text}
        
        CHRONOLOGY FORMAT EXAMPLE:
        [YYYY-MM-DD] Provider/Facility - Visit Type
        • Patient: [Name]
        • Diagnosis: [Condition]
        • Treatment: [Details]
        • Findings: [Results/Notes]
        • Follow-up: [Plan]
        
        Now generate the complete medical chronology:
        """
        
        logger.info("Generating formatted medical chronology...")
        result = self.qa_chain({"question": chronology_prompt, "chat_history": []})
        
        # Step 3: Add a summary section
        summary_prompt = """
        Based on the medical chronology, provide a concise summary that includes:
        1. Key medical conditions identified
        2. Major treatments or procedures performed
        3. Significant changes in condition
        4. Any ongoing treatment plans
        5. Overall assessment of the medical case
        """
        
        summary = self.qa_chain({"question": summary_prompt, "chat_history": []})
        
        return f"""MEDICAL CHRONOLOGY SUMMARY
{'='*80}
{summary['answer']}

DETAILED CHRONOLOGY
{'='*80}
{result['answer']}
"""
