"""Configuration settings for the Legal Document Analysis Pipeline."""
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / '.env')

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [INPUT_DIR, PROCESSED_DIR, DATABASE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/legal_docs.db"

# File processing settings
SUPPORTED_EXTENSIONS = {
    # Document formats
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.txt': 'text/plain',
    '.rtf': 'application/rtf',
    
    # Presentation formats
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.ppt': 'application/vnd.ms-powerpoint',
    
    # Spreadsheet formats
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.csv': 'text/csv',
    
    # Email formats
    '.msg': 'application/vnd.ms-outlook',
    '.eml': 'message/rfc822',
    
    # Image formats (for OCR)
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tiff': 'image/tiff',
    '.bmp': 'image/bmp',
}

# OCR settings
TESSERACT_CONFIG = {
    'path': '/usr/bin/tesseract',  # Default path, will be updated based on OS
    'languages': ['eng'],
    'oem': 3,  # OCR Engine Mode: 3 = Default
    'psm': 6,  # Page Segmentation Mode: 6 = Assume a single uniform block of text
}

# Document processing settings
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
MAX_PAGES_FOR_OCR = 50  # Maximum number of pages to process with OCR per document

# AI/ML settings
SPACY_MODEL = "en_core_web_md"  # Medium model for better entity recognition

# OpenAI Settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'legal_doc_analysis.log',
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# Update Tesseract path based on OS
if os.name == 'nt':  # Windows
    TESSERACT_CONFIG['path'] = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'
elif os.name == 'posix':  # macOS/Linux
    TESSERACT_CONFIG['path'] = '/usr/local/bin/tesseract'
    # Fallback to Homebrew location on macOS
    if not os.path.exists(TESSERACT_CONFIG['path']):
        TESSERACT_CONFIG['path'] = '/opt/homebrew/bin/tesseract'
