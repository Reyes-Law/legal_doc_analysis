"""Document ingestion module for the Legal Document Analysis Pipeline.

This module handles file system monitoring, zip file extraction, and initial
processing of incoming documents.
"""

from .file_watcher import FileWatcher
from .zip_processor import process_zip_file

__all__ = ['FileWatcher', 'process_zip_file']
