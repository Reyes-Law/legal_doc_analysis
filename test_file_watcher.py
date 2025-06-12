import time
from pathlib import Path
from legal_doc_analysis.document_ingestion import FileWatcher
from legal_doc_analysis.utils.logging import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def process_zip_file(zip_path: Path):
    """Example callback function that gets called when a new zip file is detected."""
    logger.info(f"Processing zip file: {zip_path}")
    # Here you would add your actual processing logic
    # For now, we'll just log the file details
    logger.info(f"File size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info(f"File modified: {time.ctime(zip_path.stat().st_mtime)}")

def main():
    # Create a file watcher with the default input directory
    watcher = FileWatcher()
    
    try:
        print(f"Watching directory: {watcher.watch_dir}")
        print("Press Ctrl+C to stop watching...")
        watcher.start(process_zip_file)
    except KeyboardInterrupt:
        print("\nStopping file watcher...")
        watcher.stop()
        print("File watcher stopped.")
    except Exception as e:
        logger.error(f"Error in file watcher: {e}", exc_info=True)
        watcher.stop()

if __name__ == "__main__":
    main()
