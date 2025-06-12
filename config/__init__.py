"""Configuration package for the Legal Document Analysis Pipeline."""

from .settings import *

__all__ = [
    'BASE_DIR', 'DATA_DIR', 'INPUT_DIR', 'PROCESSED_DIR', 'DATABASE_DIR', 'LOGS_DIR',
    'DATABASE_URL', 'SUPPORTED_EXTENSIONS', 'TESSERACT_CONFIG', 'MAX_FILE_SIZE_MB',
    'MAX_PAGES_FOR_OCR', 'SPACY_MODEL', 'LOGGING_CONFIG'
]
