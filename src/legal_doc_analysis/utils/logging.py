"""Logging configuration for the Legal Document Analysis Pipeline."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

from config import settings


def setup_logging(log_file: Optional[Path] = None, log_level: str = 'INFO'):
    """Configure logging for the application.
    
    Args:
        log_file: Path to the log file. If None, logs will only go to console.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_level = log_level.upper()
    
    # Define log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create formatters
    formatter = {
        'standard': {
            'format': log_format,
            'datefmt': date_format,
        },
    }
    
    # Create handlers
    handlers = {
        'console': {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
    }
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'standard',
            'filename': str(log_file),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 5,
            'encoding': 'utf8',
        }
    
    # Configure logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': formatter,
        'handlers': handlers,
        'loggers': {
            '': {  # root logger
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': True
            },
            'legal_doc_analysis': {
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': False
            },
            'fastapi': {
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': False
            },
        },
    })
    
    # Set log level for common noisy loggers
    logging.getLogger('watchdog').setLevel('WARNING')
    logging.getLogger('filelock').setLevel('WARNING')
    logging.getLogger('PIL').setLevel('WARNING')
    logging.getLogger('pdfminer').setLevel('WARNING')
    logging.getLogger('urllib3').setLevel('WARNING')
    
    # Log successful configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    return logger
