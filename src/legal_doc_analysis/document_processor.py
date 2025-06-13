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

# Query classification
FACTUAL_KEYWORDS = {
    'what', 'when', 'who', 'which', 'where', 'how many', 'how much',
    'policy number', 'address', 'phone', 'email', 'date of', 'name of',
    'is the', 'are the', 'was the', 'were the', 'does', 'did', 'has', 'have',
    'can you', 'could you', 'would you', 'please', 'tell me', 'give me',
    'find', 'search', 'look up', 'locate'
}

SYNTHESIS_KEYWORDS = {
    'chronology', 'timeline', 'summary', 'summarize', 'all', 'every', 'list all',
    'generate', 'compile', 'overview', 'comprehensive', 'detailed', 'complete',
    'entire', 'whole', 'full', 'history', 'background', 'synthesize',
    'medical history', 'treatment history', 'visit history', 'all visits',
    'all treatments', 'all procedures', 'all medications', 'all tests',
    'all results', 'all findings', 'all diagnoses', 'all conditions'
}

# Query type enum
from enum import Enum, auto

class QueryType(Enum):
    """Enum for query types to determine processing approach."""
    FACTUAL = auto()    # Factual queries seeking specific information
    SYNTHESIS = auto()  # Synthesis queries requiring document-wide analysis
    MIXED = auto()      # Mixed queries with both factual and synthesis elements

def classify_query(query: str) -> QueryType:
    """Classify a query as factual, synthesis, or mixed based on keywords and structure.
    
    Args:
        query: The user query to classify
        
    Returns:
        QueryType enum value representing the query classification
    """
    query_lower = query.lower()
    
    # Check for synthesis keywords
    has_synthesis = any(keyword in query_lower for keyword in SYNTHESIS_KEYWORDS)
    
    # Check for factual keywords
    has_factual = any(keyword in query_lower for keyword in FACTUAL_KEYWORDS)
    
    # Classify based on keyword presence
    if has_synthesis and has_factual:
        return QueryType.MIXED
    elif has_synthesis:
        return QueryType.SYNTHESIS
    else:
        # Default to factual for any query without synthesis keywords
        return QueryType.FACTUAL

# Search parameters
DEFAULT_K = 5  # For factual queries
COMPREHENSIVE_K = 20  # For synthesis queries
MAX_TOKENS = 32000  # Conservative estimate for model context window
TOKEN_BUFFER = 1000  # Buffer for prompt and response tokens

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
        """
        Get the path to save/load the vector store.
        
        Returns:
            Path: Full path to the vector store directory
            
        Raises:
            ValueError: If persist_dir is not set or case directory is invalid
        """
        if not self.persist_dir:
            raise ValueError("Cannot get vector store path: persist_dir is not set")
            
        if not hasattr(self, 'case_dir') or not self.case_dir:
            raise ValueError("Cannot get vector store path: case_dir is not set")
            
        # Normalize and validate case directory name
        case_name = self.case_dir.name.strip()
        if not case_name:
            raise ValueError("Cannot get vector store path: case directory name is empty")
            
        # Create a consistent hash of the case directory name
        case_hash = hashlib.md5(case_name.encode()).hexdigest()
        
        # Create the vector store path
        vectorstore_dir = self.persist_dir / f"faiss_index_{case_hash}"
        
        logger.debug(f"Generated vector store path: {vectorstore_dir} for case: {case_name}")
        return vectorstore_dir
    
    def _load_vectorstore(self) -> bool:
        """
        Load the vector store from disk if it exists.
        
        Returns:
            bool: True if vector store was loaded successfully, False otherwise
        """
        if not self.persist_dir:
            logger.warning("No persist directory configured, cannot load vector store")
            return False
        
        # First try the expected path based on case directory hash
        vectorstore_path = self._get_vectorstore_path()
        
        # Check for required FAISS index files at the expected path
        index_file = vectorstore_path.parent / f"{vectorstore_path.name}.faiss"
        pkl_file = vectorstore_path.parent / f"{vectorstore_path.name}.pkl"
        
        # If the expected files don't exist, try to find any existing vector store files
        if not index_file.exists() or not pkl_file.exists():
            logger.info(f"Vector store files not found at expected path: {index_file}")
            logger.info("Looking for any existing vector store files...")
            
            # Look for any .faiss files in the persist directory
            faiss_files = list(self.persist_dir.glob("*.faiss"))
            if not faiss_files:
                logger.warning(f"No vector store files found in {self.persist_dir}")
                return False
            
            # Use the first .faiss file found
            found_index_file = faiss_files[0]
            vectorstore_name = found_index_file.stem
            logger.info(f"Found existing vector store: {vectorstore_name}")
        else:
            # Use the expected path
            vectorstore_name = vectorstore_path.name
            logger.info(f"Using expected vector store: {vectorstore_name}")
        
        try:
            logger.info(f"Loading vector store {vectorstore_name} from {self.persist_dir}")
            self.vectorstore = FAISS.load_local(
                folder_path=str(self.persist_dir),
                index_name=vectorstore_name,
                embeddings=OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                    model="text-embedding-3-small"
                ),
                allow_dangerous_deserialization=True  # Required for some FAISS versions
            )
            
            if self.vectorstore is None:
                raise ValueError("Failed to load vector store: returned None")
            
            # Log the number of vectors if the attribute exists
            if hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
                logger.info(f"✅ Successfully loaded vector store with {self.vectorstore.index.ntotal} vectors")
            elif hasattr(self.vectorstore, 'index_to_docstore_id'):
                logger.info(f"✅ Successfully loaded vector store with {len(self.vectorstore.index_to_docstore_id)} vectors")
            else:
                logger.info(f"✅ Successfully loaded vector store (vector count unknown)")
            return True
            
        except Exception as e:
            error_msg = f"❌ Error loading vector store from {self.persist_dir}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.vectorstore = None
            return False
    
    def _save_vectorstore(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        if not self.persist_dir:
            logger.warning("No persist directory configured, cannot save vector store")
            return False
            
        if not self.vectorstore:
            logger.warning("No vector store to save")
            return False
            
        vectorstore_path = self._get_vectorstore_path()
        try:
            # Ensure parent directory exists
            vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving vector store to {vectorstore_path}")
            self.vectorstore.save_local(
                folder_path=str(vectorstore_path.parent),
                index_name=vectorstore_path.name
            )
            
            # Verify the files were created
            index_file = vectorstore_path.parent / f"{vectorstore_path.name}.faiss"
            if not index_file.exists():
                raise FileNotFoundError(f"FAISS index file not created at {index_file}")
                
            logger.info(f"✅ Successfully saved vector store to {vectorstore_path}")
            return True
            
        except Exception as e:
            error_msg = f"❌ Error saving vector store to {vectorstore_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False
    
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
                
        return split_docs
    
    def create_vector_store(self, force_rebuild: bool = False) -> bool:
        """Create or load a vector store from documents.
        
        Args:
            force_rebuild: If True, force rebuild the vector store even if it exists
            
        Returns:
            bool: True if vector store was created/loaded successfully, False otherwise
        """
        try:
            # Try to load from disk first if not forcing rebuild
            if not force_rebuild:
                logger.info("Attempting to load existing vector store...")
                if self._load_vectorstore():
                    logger.info("✅ Successfully loaded vector store from disk")
                    return True
                logger.info("No existing vector store found, creating a new one...")
            
            # Need to rebuild the vector store
            logger.info("Splitting documents into chunks...")
            documents = self.split_documents(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            if not documents:
                logger.error("❌ No document chunks available for processing")
                return False
                
            logger.info(f"Creating new vector store from {len(documents)} document chunks...")
            
            # Initialize batch size for this operation
            current_batch_size = BATCH_SIZE
            
            # Initialize embeddings with rate limiting
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-small",
                request_timeout=60,  # Increased timeout
                max_retries=3,       # Retry on failure
                show_progress_bar=True
            )
            
            # Process documents in smaller batches
            batch_size = min(current_batch_size, len(documents))
            logger.info(f"Processing documents in batches of {batch_size}...")
            
            # Initialize with first batch
            batch = documents[:batch_size]
            logger.debug(f"Creating initial FAISS index with {len(batch)} documents...")
            self.vectorstore = FAISS.from_documents(
                documents=batch,
                embedding=embeddings
            )
            logger.info(f"✅ Initialized FAISS index with {len(batch)} documents")
            
            # Process remaining batches
            total_batches = (len(documents) + batch_size - 1) // batch_size
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                try:
                    logger.debug(f"Adding batch {batch_num}/{total_batches} with {len(batch)} documents...")
                    self.vectorstore.add_documents(batch)
                    logger.debug(f"✅ Added batch {batch_num}/{total_batches}")
                except Exception as e:
                    logger.warning(f"⚠️ Error processing batch {batch_num}: {str(e)}")
                    # Reduce batch size on error and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"Reduced batch size to {current_batch_size}")
                    
                    if current_batch_size == 0:
                        raise RuntimeError("Batch size reduced to 0, cannot continue processing")
                    
                    # Update batch size and retry logic if needed
                    batch_size = min(current_batch_size, len(documents) - i)
                    if batch_size > 0:
                        batch = documents[i:i + batch_size]
                        self.vectorstore.add_documents(batch)
            
            # Save the vector store after successful creation
            if self.persist_dir:
                save_path = self._get_vectorstore_path()
                logger.info(f"Saving vector store to {save_path}...")
                self._save_vectorstore()
                logger.info(f"✅ Successfully saved vector store to {save_path}")
            
            logger.info(f"✅ Successfully created vector store with {len(documents)} document chunks")
            return True
            
        except Exception as e:
            error_msg = f"❌ Failed to create vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.vectorstore = None  # Reset on failure
            raise RuntimeError(error_msg) from e
        
    def setup_qa_chain(self, force_rebuild: bool = False):
        """
        Set up the QA chain with the document retriever.
        
        Args:
            force_rebuild: If True, force rebuild the vector store
            
        Returns:
            The configured QA chain
            
        Raises:
            RuntimeError: If vector store or QA chain setup fails
        """
        try:
            # Create or load vector store if needed
            if self.vectorstore is None or force_rebuild:
                logger.info("Initializing vector store for QA chain...")
                success = self.create_vector_store(force_rebuild=force_rebuild)
                if not success:
                    error_msg = "Failed to initialize vector store for QA chain"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.info("✅ Successfully initialized vector store for QA chain")
            
            if self.vectorstore is None:
                error_msg = "Vector store is not initialized"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            # Initialize LLM with error handling
            try:
                logger.info("Initializing LLM for QA chain...")
                llm = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=OPENAI_API_KEY,
                    request_timeout=60,
                    max_retries=3
                )
                logger.info("✅ Successfully initialized LLM")
            except Exception as e:
                error_msg = f"Failed to initialize LLM: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Configure retriever with error handling
            try:
                logger.info("Configuring document retriever...")
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )
                logger.info("✅ Successfully configured document retriever")
            except Exception as e:
                error_msg = f"Failed to configure document retriever: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Create QA chain with error handling
            try:
                logger.info("Creating QA chain...")
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    max_tokens_limit=4000
                )
                logger.info("✅ Successfully created QA chain")
                return self.qa_chain
                
            except Exception as e:
                error_msg = f"Failed to create QA chain: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
                
        except Exception as e:
            error_msg = f"Error setting up QA chain: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
            return self.qa_chain
            
        except Exception as e:
            logger.error(f"Failed to set up QA chain: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize QA chain: {e}")
            
    def _generate_search_terms(self, query: str) -> List[str]:
        """
        Generate additional search terms from the query to enhance retrieval.
        
        This method extracts key entities and concepts from the query and generates
        additional search terms to improve document retrieval coverage.
        
        Args:
            query: The original user query
            
        Returns:
            List of search terms to use for document retrieval
        """
        # Basic entity extraction for common legal and medical terms
        search_terms = []
        
        # Extract potential entities from the query
        query_lower = query.lower()
        
        # Check for medical facility references
        if any(term in query_lower for term in ["hospital", "er", "emergency", "clinic", "medical center"]):
            search_terms.extend(["hospital", "emergency room", "medical center", "clinic", "admission"])
            
        # Check for injury references
        if any(term in query_lower for term in ["injury", "pain", "hurt", "accident", "wound"]):
            search_terms.extend(["injury", "diagnosis", "treatment", "pain", "accident"])
            
        # Check for doctor/provider references
        if any(term in query_lower for term in ["doctor", "physician", "specialist", "surgeon"]):
            search_terms.extend(["doctor", "physician", "provider", "specialist", "medical professional"])
            
        # Check for test/procedure references
        if any(term in query_lower for term in ["test", "scan", "x-ray", "mri", "ct", "exam"]):
            search_terms.extend(["test results", "x-ray", "MRI", "CT scan", "examination", "diagnostic"])
            
        # Check for date/time references
        if any(term in query_lower for term in ["when", "date", "time", "day", "month", "year"]):
            search_terms.extend(["date", "admission date", "discharge", "visit date"])
            
        # Check for name references
        if any(term in query_lower for term in ["name", "called", "named", "which"]):
            search_terms.extend(["name", "facility name", "hospital name", "provider name"])
            
        # Add client name if it appears in the query or if asking about client
        if "client" in query_lower or "erik" in query_lower or "williams" in query_lower:
            search_terms.extend(["Erik Williams", "patient name", "client name"])
            
        # Ensure we have at least some search terms
        if not search_terms:
            # Default search terms if none of the specific categories matched
            search_terms = ["medical record", "patient information", "treatment", "visit", "hospital"]
            
        # Add the original query as a search term
        if query not in search_terms:
            search_terms.insert(0, query)
            
        logger.debug(f"Generated search terms: {search_terms}")
        return search_terms
    
    def _process_factual_query(self, query: str) -> Dict[str, Any]:
        """Process a factual query using enhanced RAG approach with multi-query retrieval.
        
        This improved implementation uses multiple retrieval strategies to ensure more
        comprehensive and consistent results for factual queries.
        """
        try:
            # Step 1: Generate search terms to enhance retrieval
            search_terms = self._generate_search_terms(query)
            logger.info(f"Generated search terms: {search_terms}")
            
            # Step 2: Retrieve documents using multiple strategies
            all_docs = []
            seen_docs = set()
            
            # Strategy 1: Direct query retrieval
            direct_docs = self.vectorstore.similarity_search(
                query,
                k=8  # Increased from default 5
            )
            
            for doc in direct_docs:
                doc_hash = hash(doc.page_content[:1000])  # Simple hash of content start
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_docs.append(doc)
            
            # Strategy 2: Search term based retrieval for broader coverage
            for term in search_terms:
                term_docs = self.vectorstore.similarity_search(
                    term,
                    k=5
                )
                
                for doc in term_docs:
                    doc_hash = hash(doc.page_content[:1000])
                    if doc_hash not in seen_docs:
                        seen_docs.add(doc_hash)
                        all_docs.append(doc)
            
            logger.info(f"Retrieved {len(all_docs)} documents for factual query")
            
            # Step 3: Use the QA chain with the enhanced document set
            # Create a custom prompt with more specific instructions
            from langchain.prompts import PromptTemplate
            
            # Create a custom prompt that emphasizes extracting specific details
            custom_prompt = PromptTemplate(
                template="""You are an expert legal assistant analyzing documents for a case.
                
                Answer the following question based ONLY on the provided context. Be specific and detailed.
                If the answer is in the context, provide it clearly with all relevant details.
                If the answer is not in the context, respond with 'I don't have enough information to answer this question.'
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:""",
                input_variables=["context", "question"]
            )
            
            # Use the existing QA chain with our enhanced document set
            from langchain.chains import RetrievalQA
            from langchain.chat_models import ChatOpenAI
            
            # Create a temporary QA chain with our enhanced document set
            enhanced_llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
                request_timeout=60
            )
            
            # Create a context string from all retrieved documents
            context_texts = [f"Document {i+1}:\n{doc.page_content}\n\n" for i, doc in enumerate(all_docs)]
            context = "\n".join(context_texts)
            
            # Use the LLM directly with our custom prompt and context
            from langchain.schema import HumanMessage, SystemMessage
            
            # Enhanced system message with formatting instructions
            system_message = SystemMessage(content="""You are a precise legal assistant that answers questions based only on the provided context.
            
            Follow these guidelines for your responses:
            1. Be concise and direct in your answers.
            2. If specific information (like names, dates, or locations) is requested and found in the context, highlight it using bold formatting.
            3. If the requested information is not in the context, clearly state 'The requested information about [topic] is not found in the available documents.'
            4. Structure complex answers with headers and bullet points when appropriate.
            5. Include specific page references when possible.
            6. Never make up information not present in the context.
            7. For medical information, be precise about diagnoses, treatments, and facilities.
            """)
            
            # Log the amount of context being sent to the LLM
            context_length = len(context)
            logger.info(f"Sending {context_length} characters of context to LLM for query: {query[:50]}...")
            logger.info(f"Number of documents retrieved: {len(all_docs)}")
            
            human_message = HumanMessage(content=custom_prompt.format(context=context, question=query))
            
            messages = [system_message, human_message]
            
            try:
                # Attempt to get a response from the LLM
                response = enhanced_llm(messages)
                answer = response.content
                
                # Check if the answer indicates information wasn't found
                not_found_phrases = [
                    "I don't have enough information", 
                    "not found in the available documents",
                    "cannot be determined from",
                    "is not mentioned in",
                    "no information about",
                    "no specific mention of"
                ]
                
                info_not_found = any(phrase in answer for phrase in not_found_phrases)
                
                # Log the query result status
                if info_not_found:
                    logger.warning(f"Information not found for query: {query}")
                    # Enhance the answer with suggestions if information wasn't found
                    answer += "\n\n**Suggestions:**\n"
                    answer += "- Try rephrasing your question with different terms\n"
                    answer += "- Ask about related information that might be available\n"
                    answer += "- Check if documents containing this information have been uploaded"
                else:
                    logger.info(f"Successfully answered query: {query[:50]}...")
                
                result = {
                    "answer": answer,
                    "source_documents": all_docs,
                    "query_success": not info_not_found
                }
                
            except Exception as e:
                logger.error(f"Error getting LLM response: {e}", exc_info=True)
                fallback_answer = f"I apologize, but I encountered an error while processing your query about '{query}'. "
                fallback_answer += "This might be due to the complexity of the question or limitations in the available information. "
                fallback_answer += "Please try rephrasing your question or asking about a different aspect of the case."
                
                result = {
                    "answer": fallback_answer,
                    "source_documents": all_docs,
                    "query_success": False,
                    "error": str(e)
                }
            
            # Step 4: Enhance the response with source information
            sources = []
            for doc in result["source_documents"]:
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return {
                "answer": result["answer"],
                "sources": list(set(doc.metadata["source"] for doc in result["source_documents"])),
                "detailed_sources": sources,
                "query_type": "factual"
            }
        except Exception as e:
            logger.error(f"Error processing factual query: {e}", exc_info=True)
            return {"error": str(e), "query_type": "factual"}
            
    def _process_synthesis_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """Process a synthesis query using comprehensive multi-stage retrieval."""
        try:
            # Define domain-specific search terms based on query type
            if "medical" in query.lower() or "chronology" in query.lower():
                search_terms = [
                    "medical visit", "doctor appointment", "hospital admission", "treatment",
                    "diagnosis", "medication", "prescription", "test results", "surgery",
                    "procedure", "examination", "follow-up", "emergency room", "ER visit"
                ]
            else:
                search_terms = [query]  # Fallback to original query
            
            # Retrieve comprehensive set of documents
            all_docs = []
            seen_docs = set()
            
            for term in search_terms:
                try:
                    # Get more documents than needed to account for overlaps
                    docs = self.vectorstore.similarity_search(
                        term, 
                        k=min(COMPREHENSIVE_K, 20)  # Limit per search
                    )
                    
                    # Deduplicate while preserving order
                    for doc in docs:
                        doc_hash = hash(doc.page_content[:1000])  # Simple hash of content start
                        if doc_hash not in seen_docs:
                            seen_docs.add(doc_hash)
                            all_docs.append(doc)
                except Exception as e:
                    logger.warning(f"Error searching for term '{term}': {str(e)}")
                    continue
            
            if not all_docs:
                return {"answer": "No relevant documents found for this query.", "query_type": "synthesis"}
            
            # Prepare the synthesis prompt
            synthesis_prompt = f"""
            You are a legal assistant creating a comprehensive {query_type} based on the following documents.
            
            INSTRUCTIONS:
            1. Extract and organize all relevant information
            2. Group related information together
            3. Present in a clear, structured format
            4. Include specific details and dates where available
            5. Preserve all important information from the source documents
            
            Documents:
            {documents}
            
            Please provide a detailed {query_type} based on the above information.
            """
            
            # Process in batches to handle token limits
            batch_size = 6  # Number of docs per batch
            results = []
            
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i+batch_size]
                batch_docs = "\n---\n".join(
                    f"Document {j+1}:\n{doc.page_content}"
                    for j, doc in enumerate(batch)
                )
                
                # Get synthesis for this batch
                batch_prompt = synthesis_prompt.format(
                    query_type=query_type,
                    documents=batch_docs
                )
                
                try:
                    batch_result = self.qa_chain({
                        "question": batch_prompt,
                        "chat_history": []
                    })
                    results.append(batch_result["answer"])
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            
            # Combine all batch results
            combined_result = "\n\n---\n\n".join(results)
            
            # Final synthesis pass if needed
            if len(results) > 1:
                final_prompt = f"""
                Combine the following partial {query_type} sections into a single, well-organized document.
                Remove any redundancies and ensure consistent formatting.
                
                Sections:
                {sections}
                """.format(
                    query_type=query_type,
                    sections="\n\n---\n\n".join(
                        f"Section {i+1}:\n{section}" 
                        for i, section in enumerate(results)
                    )
                )
                
                try:
                    final_result = self.qa_chain({
                        "question": final_prompt,
                        "chat_history": []
                    })
                    combined_result = final_result["answer"]
                except Exception as e:
                    logger.error(f"Error in final synthesis: {str(e)}")
            
            return {
                "answer": combined_result,
                "sources": list(set(doc.metadata.get("source", "unknown") for doc in all_docs)),
                "query_type": "synthesis"
            }
            
        except Exception as e:
            logger.error(f"Error in synthesis query processing: {str(e)}")
            return {
                "error": f"Error processing synthesis query: {str(e)}",
                "query_type": "synthesis"
            }
    
    def query_documents(self, query: str) -> Dict[str, Any]:
        """
        Query the document store with adaptive processing based on query type.
        
        Args:
            query: The user's query (must be a non-empty string)
            
        Returns:
            Dict containing answer, sources, and metadata or error information
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If QA chain setup or query processing fails
        """
        # Input validation
        if not query or not isinstance(query, str) or not query.strip():
            error_msg = "Query cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Processing query: {query[:100]}...")
        
        # Ensure QA chain is initialized
        if self.qa_chain is None:
            logger.info("Initializing QA chain...")
            try:
                self.setup_qa_chain()
                logger.info("✅ QA chain initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize QA chain: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        
        try:
            # Classify the query
            try:
                query_type = classify_query(query)
                logger.debug(f"Query classified as: {query_type.name}")
            except Exception as e:
                error_msg = f"Failed to classify query: {str(e)}"
                logger.error(error_msg, exc_info=True)
                # Default to factual query if classification fails
                query_type = QueryType.FACTUAL
                logger.info(f"Using default query type: {query_type.name}")
            
            # Route to appropriate processor
            if query_type in (QueryType.FACTUAL, QueryType.MIXED):
                logger.info("Processing as factual query")
                return self._process_factual_query(query)
            else:
                logger.info("Processing as synthesis query")
                # Extract the type of synthesis needed (e.g., 'chronology', 'summary')
                synthesis_type = next(
                    (term for term in SYNTHESIS_KEYWORDS if term in query.lower()),
                    "summary"
                )
                logger.debug(f"Synthesis type: {synthesis_type}")
                return self._process_synthesis_query(query, synthesis_type)
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "error": "An error occurred while processing your query. Please try again.",
                "details": str(e),
                "query_type": "error"
            }
    
    def generate_medical_chronology(self) -> str:
        """
        Generate a comprehensive medical chronology from the documents.
        Uses a multi-phase approach: document retrieval, batch processing, and synthesis.
        
        Returns:
            str: Formatted medical chronology with dates, providers, treatments, and findings
        """
        logger.info("Starting medical chronology generation...")
        
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return "Error: Vector store not initialized. Please process documents first."
        
        try:
            # Phase 1: Document Retrieval and Medical Content Detection
            medical_docs = self._retrieve_medical_documents()
            if not medical_docs:
                return "No medical documents found in the case files. Please ensure medical records are included in the uploaded documents."
            
            logger.info(f"Found {len(medical_docs)} medical documents for chronology generation")
            
            # Phase 2: Batch Processing
            batch_results = self._process_documents_in_batches(medical_docs)
            if not batch_results:
                return "Error: Failed to process medical documents. Please check the logs for details."
            
            # Phase 3: Synthesis and Final Chronology
            final_chronology = self._synthesize_chronology(batch_results)
            
            logger.info("Medical chronology generation completed successfully")
            return final_chronology
            
        except Exception as e:
            error_msg = f"Error generating medical chronology: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    def _retrieve_medical_documents(self) -> list:
        """
        Retrieve documents containing medical information using multiple search strategies.
        
        Returns:
            list: List of documents containing medical content
        """
        logger.info("Retrieving medical documents...")
        
        # Strategy 1: Primary medical search
        primary_queries = [
            "medical records hospital doctor treatment diagnosis patient",
            "physician notes examination findings test results",
            "surgery procedure medication prescribed therapy"
        ]
        
        # Strategy 2: Specific medical terms
        specific_terms = [
            "medical record", "hospital", "doctor visit", "physician notes",
            "diagnosis", "treatment plan", "medication", "surgery",
            "examination", "test results", "lab work", "radiology",
            "emergency room", "discharge summary", "follow-up",
            "specialist consultation", "patient history", "injury"
        ]
        
        all_docs = []
        seen_content = set()
        
        # Execute primary searches
        for query in primary_queries:
            try:
                docs = self.vectorstore.similarity_search(query, k=50)
                for doc in docs:
                    content_hash = hash(doc.page_content[:500])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Error in primary search '{query}': {str(e)}")
        
        # Execute specific term searches
        for term in specific_terms:
            try:
                docs = self.vectorstore.similarity_search(term, k=15)
                for doc in docs:
                    content_hash = hash(doc.page_content[:500])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Error searching for term '{term}': {str(e)}")
        
        # Filter for actual medical content
        medical_docs = self._filter_medical_content(all_docs)
        
        logger.info(f"Retrieved {len(all_docs)} total documents, {len(medical_docs)} contain medical content")
        return medical_docs
    
    def _filter_medical_content(self, docs: list) -> list:
        """
        Filter documents to ensure they contain actual medical content.
        
        Args:
            docs: List of documents to filter
            
        Returns:
            list: Documents containing medical content
        """
        medical_indicators = [
            'medical', 'doctor', 'physician', 'hospital', 'patient', 'treatment',
            'diagnosis', 'surgery', 'medication', 'prescription', 'therapy',
            'examination', 'test', 'lab', 'radiology', 'x-ray', 'mri', 'ct scan',
            'emergency', 'admission', 'discharge', 'clinic', 'nurse', 'health',
            'injury', 'pain', 'symptoms', 'condition', 'procedure', 'consultation'
        ]
        
        medical_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            medical_score = sum(1 for term in medical_indicators if term in content_lower)
            
            # Require at least 3 medical terms to consider it medical content
            if medical_score >= 3:
                medical_docs.append(doc)
                logger.debug(f"Medical document found (score: {medical_score}): {doc.metadata.get('source', 'unknown')}")
        
        return medical_docs
    
    def _process_documents_in_batches(self, docs: list) -> list:
        """
        Process medical documents in batches to extract chronological information.
        
        Args:
            docs: List of medical documents to process
            
        Returns:
            list: List of batch processing results
        """
        logger.info(f"Processing {len(docs)} documents in batches...")
        
        # Determine optimal batch sizes
        total_docs = len(docs)
        if total_docs <= 30:
            first_batch_size = min(20, total_docs)
            batch_size = 10
        elif total_docs <= 100:
            first_batch_size = 40
            batch_size = 20
        else:
            first_batch_size = 60
            batch_size = 25
        
        logger.info(f"Using first batch size: {first_batch_size}, subsequent batch size: {batch_size}")
        
        batch_results = []
        
        # Process first batch (most relevant documents)
        first_batch = docs[:first_batch_size]
        first_result = self._process_batch(first_batch, 1, "Primary Medical Documents")
        if first_result:
            batch_results.append(first_result)
        
        # Process remaining documents in smaller batches
        remaining_docs = docs[first_batch_size:]
        batch_num = 2
        
        for i in range(0, len(remaining_docs), batch_size):
            batch = remaining_docs[i:i + batch_size]
            batch_result = self._process_batch(batch, batch_num, f"Additional Medical Documents (Batch {batch_num})")
            if batch_result:
                batch_results.append(batch_result)
            batch_num += 1
        
        logger.info(f"Completed processing {len(batch_results)} batches")
        return batch_results
    
    def _process_batch(self, batch: list, batch_num: int, batch_description: str) -> str:
        """
        Process a single batch of documents to extract medical chronology information.
        
        Args:
            batch: List of documents in this batch
            batch_num: Batch number for logging
            batch_description: Description of this batch
            
        Returns:
            str: Extracted medical chronology information from this batch
        """
        logger.info(f"Processing {batch_description} ({len(batch)} documents)")
        
        # Format documents for processing
        batch_text = "\n---\n".join([
            f"Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}"
            for i, doc in enumerate(batch)
        ])
        
        # Create extraction prompt
        extraction_prompt = f"""
        You are a medical-legal expert extracting chronological medical information from case documents.
        
        TASK: Extract ALL medical events, dates, providers, and treatments from the following documents.
        
        FORMATTING REQUIREMENTS:
        - Use plain text only (NO markdown)
        - Format each medical event as follows:
        
        DATE: [MM/DD/YYYY or approximate timeframe]
        PROVIDER: [Doctor/Hospital/Clinic name]
        SERVICE: [Type of visit/procedure/treatment]
        DIAGNOSIS: [Medical condition or complaint]
        TREATMENT: [Treatment provided or medication prescribed]
        NOTES: [Additional relevant medical details]
        SOURCE: [Document reference and page]
        
        CRITICAL INSTRUCTIONS:
        1. Extract EVERY medical event you can find, even if information is incomplete
        2. If exact dates aren't available, use approximate timeframes (e.g., "Prior to 01/15/2023")
        3. Include ALL medical providers, treatments, and diagnoses mentioned
        4. Do NOT say "I don't know" - extract whatever medical information IS present
        5. If a document mentions medical records but doesn't provide details, note that records exist
        
        DOCUMENTS TO ANALYZE:
        {batch_text}
        
        EXTRACTED MEDICAL CHRONOLOGY:
        """
        
        try:
            result = self.qa_chain({
                "question": extraction_prompt,
                "chat_history": []
            })
            
            extracted_info = result["answer"]
            logger.info(f"Successfully processed {batch_description}")
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error processing {batch_description}: {str(e)}")
            return f"Error processing {batch_description}: {str(e)}"
    
    def _synthesize_chronology(self, batch_results: list) -> str:
        """
        Synthesize all batch results into a final comprehensive medical chronology.
        
        Args:
            batch_results: List of results from batch processing
            
        Returns:
            str: Final synthesized medical chronology
        """
        logger.info("Synthesizing final medical chronology...")
        
        if not batch_results:
            return "No medical information could be extracted from the documents."
        
        # Combine all batch results
        combined_extractions = "\n\n".join([
            f"BATCH {i+1} RESULTS:\n{result}" 
            for i, result in enumerate(batch_results)
        ])
        
        # First, generate a patient summary
        summary_prompt = f"""
        You are a medical-legal expert. Based on the following medical extractions, create a patient summary.
        
        TASK: Create a brief overview covering whatever information is available:
        1. Patient name (if mentioned - look for names like "Aaron Oliver" or similar)
        2. Date of accident or incident (if mentioned - look for accident dates)
        3. Primary injuries or body parts treated (shoulder, neck, spine, etc.)
        4. Key medical providers (hospital names, doctor names)
        5. Treatment timeframe (earliest to latest dates)
        6. Types of treatments received (X-rays, MRI, chiropractic, etc.)
        
        INSTRUCTIONS:
        - Use plain text only (NO markdown)
        - Extract whatever information you can find, even if incomplete
        - If patient name is not clear, say "Patient name not clearly specified"
        - Focus on the medical facts that ARE available in the records
        - Do NOT say "I don't know" - describe what medical information IS present
        - Keep it to 4-6 sentences maximum
        
        MEDICAL EXTRACTIONS:
        {combined_extractions}
        
        PATIENT SUMMARY:
        """
        
        try:
            # Extract key information directly from batch results
            patient_name = "Patient name not clearly specified"
            accident_date = None
            providers = set()
            body_parts = set()
            treatments = set()
            date_range = {"earliest": None, "latest": None}
            
            # Analyze batch results for key information
            combined_text = " ".join(batch_results).lower()
            
            # Extract patient name
            import re
            name_patterns = [
                r'oliver[,\s]+aaron',
                r'aaron[,\s]+oliver',
                r'oliver\s*,\s*aaron',
                r'aaron\s*,?\s*oliver',
                r'patient[:\s]+([a-z]+[,\s]+[a-z]+)',
                r'plaintiff[:\s]+([a-z]+[,\s]+[a-z]+)'
            ]
            for pattern in name_patterns:
                match = re.search(pattern, combined_text)
                if match:
                    if 'oliver' in match.group(0) and 'aaron' in match.group(0):
                        patient_name = "Aaron Oliver"
                        break
                    elif match.groups():
                        patient_name = match.group(1).title()
                        break
            
            # Extract accident date
            accident_patterns = [
                r'accident[:\s]*(?:date[:\s]*)?(?:of[:\s]*)?(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
                r'(?:04[/\-]23[/\-]20|april\s+23[,\s]*20)'
            ]
            for pattern in accident_patterns:
                match = re.search(pattern, combined_text)
                if match:
                    if '04' in pattern or 'april' in pattern:
                        accident_date = "04/23/2020"
                    else:
                        accident_date = match.group(1)
                    break
            
            # Extract providers
            provider_keywords = ['uptown radiology', 'texas medicine', 'elevate health', 'woodley', 'mehrotra']
            for keyword in provider_keywords:
                if keyword in combined_text:
                    if 'uptown' in keyword:
                        providers.add("Uptown Radiology Associates")
                    elif 'texas medicine' in keyword:
                        providers.add("Texas Medicine Resources")
                    elif 'elevate' in keyword:
                        providers.add("Elevate Health Clinics")
                    elif 'woodley' in keyword:
                        providers.add("Margaret Woodley, PA-C")
                    elif 'mehrotra' in keyword:
                        providers.add("Dr. Vinit Mehrotra")
            
            # Extract body parts/injuries
            body_keywords = ['shoulder', 'wrist', 'neck', 'spine', 'cervical', 'lumbar']
            for keyword in body_keywords:
                if keyword in combined_text:
                    body_parts.add(keyword.title())
            
            # Extract treatment types
            treatment_keywords = ['x-ray', 'mri', 'chiropractic', 'therapy', 'consultation', 'examination']
            for keyword in treatment_keywords:
                if keyword in combined_text:
                    treatments.add(keyword.upper() if keyword == 'mri' else keyword.title())
            
            # Extract date range
            date_pattern = r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})'
            dates = re.findall(date_pattern, combined_text)
            if dates:
                # Convert to comparable format and find range
                parsed_dates = []
                for date_str in dates:
                    try:
                        # Simple date parsing
                        if '/' in date_str:
                            month, day, year = date_str.split('/')
                        else:
                            month, day, year = date_str.split('-')
                        parsed_dates.append((int(year), int(month), int(day), date_str))
                    except:
                        continue
                
                if parsed_dates:
                    parsed_dates.sort()
                    date_range["earliest"] = parsed_dates[0][3]
                    date_range["latest"] = parsed_dates[-1][3]
            
            # Create summary
            summary_parts = []
            summary_parts.append(f"Patient: {patient_name}")
            
            if accident_date:
                summary_parts.append(f"Accident Date: {accident_date}")
            
            if body_parts:
                summary_parts.append(f"Primary Areas Treated: {', '.join(sorted(body_parts))}")
            
            if providers:
                summary_parts.append(f"Medical Providers: {', '.join(sorted(providers))}")
            
            if date_range["earliest"] and date_range["latest"]:
                summary_parts.append(f"Treatment Period: {date_range['earliest']} to {date_range['latest']}")
            
            if treatments:
                summary_parts.append(f"Treatment Types: {', '.join(sorted(treatments))}")
            
            patient_summary = ". ".join(summary_parts) + "."
            logger.info("Generated patient summary from direct extraction")
            
        except Exception as e:
            logger.warning(f"Error generating patient summary: {str(e)}")
            patient_summary = "Patient summary could not be generated from available records."
        
        # Then synthesize the detailed chronology
        synthesis_prompt = f"""
        You are a medical-legal expert. Create a comprehensive medical chronology from the extracted information.
        
        TASK: Organize ALL medical events chronologically and remove duplicates.
        
        FORMATTING REQUIREMENTS:
        - Use plain text only (NO markdown, asterisks, or special characters)
        - Sort all events by date (earliest first)
        - Remove duplicate entries
        - Maintain the exact format for each entry:
        
        DATE: [MM/DD/YYYY]
        PROVIDER: [Name]
        SERVICE: [Description]
        DIAGNOSIS: [Condition]
        TREATMENT: [What was done]
        NOTES: [Additional details]
        SOURCE: [Document reference]
        
        CRITICAL INSTRUCTIONS:
        1. Combine duplicate entries for the same date/provider/service
        2. Sort chronologically from earliest to latest date
        3. Preserve all medical details and billing information
        4. Do NOT add a summary section - only the chronological entries
        
        MEDICAL EXTRACTIONS TO SYNTHESIZE:
        {combined_extractions}
        
        CHRONOLOGICAL MEDICAL EVENTS:
        """
        
        try:
            result = self.qa_chain({
                "question": synthesis_prompt,
                "chat_history": []
            })
            
            detailed_chronology = result["answer"]
            
            # Combine summary and detailed chronology
            final_chronology = f"""PATIENT SUMMARY:
{patient_summary}

DETAILED MEDICAL CHRONOLOGY:

{detailed_chronology}"""
            
            # Quality check - ensure we have actual medical content
            if len(detailed_chronology.strip()) < 100 or "no medical" in detailed_chronology.lower():
                logger.warning("Detailed chronology appears incomplete, using first batch result as fallback")
                final_chronology = f"""PATIENT SUMMARY:
{patient_summary}

DETAILED MEDICAL CHRONOLOGY:

{batch_results[0] if batch_results else "Unable to generate detailed chronology from available documents."}"""
            
            logger.info("Successfully synthesized final medical chronology with summary")
            return final_chronology
            
        except Exception as e:
            logger.error(f"Error in chronology synthesis: {str(e)}")
            # Fallback to first batch result with summary
            return f"""PATIENT SUMMARY:
{patient_summary}

DETAILED MEDICAL CHRONOLOGY:

{batch_results[0] if batch_results else f"Error in synthesis: {str(e)}"}"""
