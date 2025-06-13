#!/usr/bin/env python3
"""
Web interface for the Legal Document Analysis System.
This FastAPI application provides a REST API and web UI for:
1. Uploading and processing legal case files
2. Querying documents for specific information
3. Generating medical chronologies
"""
import os
import sys
import logging
import shutil
import tempfile
from pathlib import Path
import time
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import FastAPI components
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any

# Import project modules
from src.legal_doc_analysis.document_processor import DocumentProcessor
from src.legal_doc_analysis.document_ingestion.zip_processor import process_zip_file
from config.settings import INPUT_DIR, PROCESSED_DIR, VECTOR_STORE_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Legal Document Analysis System",
    description="API for processing and querying legal documents",
    version="0.1.0"
)

# Create directories if they don't exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Create templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Create static directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Define API models
class QueryRequest(BaseModel):
    case_name: str
    question: Optional[str] = None
    query: Optional[str] = None
    
    @validator('question', 'query')
    def validate_question_or_query(cls, v, values):
        # Ensure at least one of question or query is provided
        if v is None and 'question' in values and 'query' in values and values['question'] is None and values['query'] is None:
            raise ValueError('Either question or query must be provided')
        return v
        
    def get_question(self) -> str:
        """Get the question from either question or query field."""
        return self.question or self.query or ''

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    processing_time: Optional[str] = None
    source_count: Optional[int] = None

class ChronologyRequest(BaseModel):
    case_name: str

class ChronologyResponse(BaseModel):
    chronology: str

# Helper functions
def process_case(zip_path: Path, force_rebuild: bool = False) -> Dict[str, Any]:
    """Process a case ZIP file and create a vector store."""
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
    """Query a processed case."""
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
        
        # Include detailed sources if available
        detailed_sources = result.get('detailed_sources', [])
        
        # Pass along query success status and other metadata
        return {
            'success': True,
            'answer': result.get('answer', 'No answer found'),
            'sources': sources,
            'query_success': result.get('query_success', True),
            'detailed_sources': detailed_sources,
            'document_count': len(result.get('source_documents', [])) if 'source_documents' in result else 0
        }
        
    except Exception as e:
        logger.error(f"Error querying case: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def generate_chronology(case_name: str) -> Dict[str, Any]:
    """Generate a medical chronology for a case."""
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
        
        # Check if the chronology is empty or just contains "I don't know"
        if not chronology or chronology.strip() == "I don't know." or "I don't know" in chronology:
            # Fallback: Use a direct approach to extract medical information
            logger.warning("Standard chronology generation failed, using fallback method")
            
            # Get all medical-related documents
            medical_docs = processor.vectorstore.similarity_search(
                "medical records hospital doctor treatment diagnosis", 
                k=50  # Get more documents
            )
            
            # Create a direct prompt for the LLM
            fallback_prompt = """
            You are a medical-legal expert. Based on the following document excerpts, create a medical chronology.
            
            INSTRUCTIONS:
            1. Extract ANY dates, medical providers, treatments, or events you can find.
            2. Format each entry with a date (if available) and description.
            3. If exact dates aren't available, use approximate timeframes.
            4. Include patient name, medical providers, and any medical details you can find.
            5. If information is very limited, simply list whatever medical facts are available.
            6. DO NOT say 'I don't know' - provide whatever limited information you can extract.
            
            DOCUMENTS:
            {}
            
            MEDICAL CHRONOLOGY:
            """.format("\n\n---\n\n".join([doc.page_content for doc in medical_docs[:30]]))
            
            try:
                # Direct LLM call for fallback
                fallback_result = processor.qa_chain({"question": fallback_prompt, "chat_history": []})
                chronology = fallback_result["answer"]
                
                # If still getting "I don't know", provide a helpful message
                if not chronology or chronology.strip() == "I don't know." or "I don't know" in chronology:
                    chronology = """Based on the available documents, a complete medical chronology could not be generated.
                    
                    The system identified some medical references but could not construct a full timeline. 
                    This may be because:
                    1. The documents contain limited medical information
                    2. Medical records may need to be processed or uploaded
                    3. The medical information is not in a format that allows for chronological organization
                    
                    Consider uploading additional medical records or using the query feature to ask specific 
                    medical questions about the case."""
            except Exception as e:
                logger.error(f"Fallback chronology generation failed: {e}")
                # If fallback fails, return original chronology
        
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
    """List all processed cases."""
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

# API routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/cases", response_model=List[Dict[str, Any]])
async def api_list_cases():
    """List all processed cases."""
    return list_cases()

@app.post("/api/cases/upload")
async def api_upload_case(file: UploadFile = File(...), force_rebuild: bool = Form(False)):
    """Upload and process a new case."""
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Save the uploaded file
    temp_file = Path(tempfile.gettempdir()) / file.filename
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the case
        result = process_case(temp_file, force_rebuild)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

@app.post("/api/query", response_model=QueryResponse)
async def api_query_case(request: QueryRequest):
    """Query a case."""
    start_time = time.time()
    
    # Get the question from either question or query field
    question = request.get_question()
    
    # Log the request for debugging and executive assessment
    logger.info(f"Query request: case_name={request.case_name}, question={question}")
    
    # Process the query
    result = query_case(request.case_name, question)
    
    # Calculate processing time for performance tracking
    processing_time = time.time() - start_time
    
    if not result['success']:
        logger.error(f"Query failed: {result['error']}")
        # Log metrics for executive assessment
        logger.error(f"Query metrics - Status: FAILED, Time: {processing_time:.2f}s, Case: {request.case_name}, Query: {question}")
        raise HTTPException(status_code=404, detail=result['error'])
    
    # Enhanced logging for executive assessment
    source_count = len(result.get('sources', []))
    query_success = result.get('query_success', True)  # Default to True if not provided
    status = "SUCCESS" if query_success else "PARTIAL" 
    
    logger.info(f"Query metrics - Status: {status}, Time: {processing_time:.2f}s, Sources: {source_count}, Case: {request.case_name}")
    
    # Check if the answer contains suggestions (indicating information wasn't found)
    contains_suggestions = "**Suggestions:**" in result.get('answer', '')
    
    # Format the response
    response = {
        'answer': result['answer'],
        'sources': result['sources'],
        'processing_time': f"{processing_time:.2f}s",
        'source_count': source_count
    }
    
    return response

@app.post("/api/chronology", response_model=ChronologyResponse)
async def api_generate_chronology(request: ChronologyRequest):
    """Generate a medical chronology for a case."""
    result = generate_chronology(request.case_name)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return {
        'chronology': result['chronology']
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
