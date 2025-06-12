"""File system watcher for monitoring new zip files in the input directory.

This module provides functionality to watch for new zip files in a specified
directory and trigger processing when they are detected.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from config import settings

logger = logging.getLogger(__name__)


class ZipFileHandler(FileSystemEventHandler):
    """Handler for new zip file creation events."""
    
    def __init__(self, callback: Callable[[Path], None]):
        """Initialize with a callback function to process new zip files."""
        self.callback = callback
        super().__init__()
    
    def on_created(self, event: FileCreatedEvent):
        """Called when a file is created in the watched directory."""
        if not event.is_directory and event.src_path.lower().endswith('.zip'):
            logger.info(f"Detected new zip file: {event.src_path}")
            self.callback(Path(event.src_path))


class FileWatcher:
    """Watches a directory for new zip files and processes them."""
    
    def __init__(self, watch_dir: Optional[Path] = None):
        """Initialize the file watcher.
        
        Args:
            watch_dir: Directory to watch for new zip files. Defaults to settings.INPUT_DIR.
        """
        self.watch_dir = Path(watch_dir) if watch_dir else settings.INPUT_DIR
        self.observer = Observer()
        self.running = False
        
        # Ensure the watch directory exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FileWatcher for directory: {self.watch_dir}")
    
    def start(self, callback: Callable[[Path], None]):
        """Start watching for new zip files.
        
        Args:
            callback: Function to call when a new zip file is detected.
                     The function should accept a single Path argument.
        """
        if self.running:
            logger.warning("FileWatcher is already running")
            return
        
        logger.info("Starting FileWatcher")
        event_handler = ZipFileHandler(callback)
        self.observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        self.running = True
        logger.info(f"Watching for new zip files in: {self.watch_dir}")
    
    def stop(self):
        """Stop watching for new zip files."""
        if not self.running:
            return
            
        logger.info("Stopping FileWatcher")
        self.observer.stop()
        self.observer.join()
        self.running = False
        logger.info("FileWatcher stopped")
    
    def run(self, callback: Callable[[Path], None]):
        """Run the file watcher in the current thread.
        
        This is a blocking call that will run until interrupted.
        
        Args:
            callback: Function to call when a new zip file is detected.
        """
        self.start(callback)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Error in FileWatcher: {e}")
            self.stop()
            raise
