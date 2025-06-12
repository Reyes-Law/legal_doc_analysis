"""Main entry point for the Legal Document Analysis Pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from config import settings
from legal_doc_analysis.utils import setup_logging, check_system_requirements, install_system_dependencies
from legal_doc_analysis.document_ingestion import FileWatcher

# Set up logging
logger = setup_logging(settings.LOGS_DIR / 'legal_doc_analysis.log')

def process_zip_file(zip_path: Path):
    """Process a zip file that was dropped into the input directory.
    
    Args:
        zip_path: Path to the zip file to process.
    """
    from legal_doc_analysis.document_ingestion.zip_processor import process_zip_file as process_zip
    from config.settings import PROCESSED_DIR
    
    try:
        logger.info(f"Processing zip file: {zip_path}")
        print(f"\n[INFO] Processing zip file: {zip_path.name}")
        
        # Process the zip file
        result = process_zip(zip_path, output_dir=PROCESSED_DIR / zip_path.stem)
        
        if result['success']:
            print(f"✅ Successfully extracted {len(result['extracted_files'])} files to: {result['output_directory']}")
            print(f"📄 Extracted files:")
            for file_path in result['extracted_files']:
                print(f"   - {Path(file_path).name}")
            
            # TODO: Add document processing logic here
            # For now, just list the extracted files
            
            return True
        else:
            print(f"❌ Failed to process {zip_path.name}: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        error_msg = f"Error processing {zip_path.name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"❌ {error_msg}")
        return False


def check_requirements() -> bool:
    """Check system requirements and return True if all requirements are met."""
    logger.info("Checking system requirements...")
    requirements = check_system_requirements()
    return print_requirement_status(requirements)


def print_requirement_status(requirements: dict) -> bool:
    """Print the status of system requirements and return True if all are met."""
    print("\n=== System Requirements Check ===\n")
    
    all_ok = True
    
    # Python version
    py_status, py_msg = requirements['python']['Python Version']
    status_str = "[PASS]" if py_status else "[FAIL]"
    print(f"Python Version: {status_str} {py_msg}")
    all_ok = all_ok and py_status
    
    # System commands
    print("\nSystem Commands:")
    for cmd, (status, msg) in requirements['system_commands'].items():
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str} {cmd}: {msg}")
        all_ok = all_ok and status
    
    # Python packages
    print("\nPython Packages:")
    for pkg, (status, msg) in requirements['python_packages'].items():
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str} {msg}")
        all_ok = all_ok and status
    
    print("\n=== Summary ===")
    if all_ok:
        print("\n[SUCCESS] All requirements are satisfied!")
    else:
        print("\n[WARNING] Some requirements are not met. Run with '--install-deps' to install missing dependencies.")
    
    return all_ok


def main():
    """Main entry point for the Legal Document Analysis Pipeline."""
    parser = argparse.ArgumentParser(description='Legal Document Analysis Pipeline')
    parser.add_argument('--check', action='store_true', help='Check system requirements')
    parser.add_argument('--install-deps', action='store_true', help='Install system dependencies')
    parser.add_argument('--watch', action='store_true', help='Watch for new zip files')
    args = parser.parse_args()
    
    try:
        if args.check:
            check_requirements()
            return 0
            
        if args.install_deps:
            if install_system_dependencies():
                print("\n[SUCCESS] Dependencies installed successfully!")
                print("Please restart the application to apply changes.")
            else:
                print("\n[ERROR] Failed to install some dependencies.")
                print("Please install them manually and try again.")
            return 0
        
        if args.watch:
            # Check requirements before starting the watcher
            if not check_requirements():
                print("\n[ERROR] System requirements not met. Please install missing dependencies first.")
                return 1
                
            print(f"\n[INFO] Watching for new zip files in: {settings.INPUT_DIR}")
            print("Press Ctrl+C to stop\n")
            
            watcher = FileWatcher()
            watcher.run(process_zip_file)
            return 0
            
        # Default action: check requirements
        return 0 if check_requirements() else 1
        
    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user")
        return 0
    except Exception as e:
        logger.exception("An error occurred:")
        print(f"\n[ERROR] {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
