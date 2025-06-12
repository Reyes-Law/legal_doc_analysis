import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
from legal_doc_analysis.utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_environment():
    """Check if the environment is properly set up."""
    print("\n=== Environment Check ===")
    
    # Check Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    required_vars = ["OPENAI_API_KEY", "INPUT_DIR", "OUTPUT_DIR", "DB_PATH", "LOG_FILE"]
    all_set = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set")
            if var == "OPENAI_API_KEY":
                print(f"   (Value: {value[:5]}...{value[-4:]})")
            else:
                print(f"   (Value: {value})")
        else:
            print(f"❌ {var}: Not set")
            all_set = False
    
    return all_set

def check_directories():
    """Check if required directories exist and are writable."""
    print("\n=== Directory Check ===")
    
    dirs_to_check = [
        "data/input",
        "data/output",
        "data/db",
        "data/logs"
    ]
    
    all_ok = True
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            print(f"✅ Directory {dir_path}: Exists and writable")
        except Exception as e:
            print(f"❌ Directory {dir_path}: Error - {str(e)}")
            all_ok = False
    
    return all_ok

def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n=== Python Packages Check ===")
    
    required_packages = [
        "fastapi", "uvicorn", "python-multipart", "python-dotenv", "watchdog",
        "pytesseract", "pdf2image", "PyPDF2", "python-pptx", "python-docx",
        "extract-msg", "spacy", "langchain", "openai", "sentence-transformers",
        "sqlalchemy"
    ]
    
    missing_packages = []
    
    # First try using pkg_resources
    try:
        import pkg_resources
        for package in required_packages:
            try:
                pkg_resources.get_distribution(package)
                print(f"✅ {package}: Installed")
            except pkg_resources.DistributionNotFound:
                print(f"❌ {package}: Not installed")
                missing_packages.append(package)
    except ImportError:
        # Fall back to __import__ if pkg_resources is not available
        print("Warning: pkg_resources not available, using fallback method")
        for package in required_packages:
            try:
                __import__(package.split('.')[0])
                print(f"✅ {package}: Installed")
            except ImportError:
                print(f"❌ {package}: Not installed")
                missing_packages.append(package)
    
    if missing_packages:
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_system_dependencies():
    """Check if required system dependencies are installed."""
    print("\n=== System Dependencies Check ===")
    
    dependencies = [
        {"name": "Tesseract OCR", "command": "tesseract", "version_flag": "--version"},
        {"name": "Poppler (pdftoppm)", "command": "pdftoppm", "version_flag": "-v"}
    ]
    
    all_ok = True
    
    for dep in dependencies:
        try:
            import subprocess
            result = subprocess.run(
                [dep["command"], dep["version_flag"]],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip() or result.stderr.strip()
            version = version.split('\n')[0]  # Get first line of version info
            print(f"✅ {dep['name']}: Found ({version})")
        except Exception as e:
            print(f"❌ {dep['name']}: Not found or error - {str(e)}")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks and report status."""
    print("\n" + "="*50)
    print("  Legal Document Analysis - Setup Verification")
    print("="*50)
    
    env_ok = check_environment()
    dirs_ok = check_directories()
    packages_ok = check_python_packages()
    deps_ok = check_system_dependencies()
    
    print("\n" + "="*50)
    print("  Setup Verification Complete")
    print("="*50)
    
    if env_ok and dirs_ok and packages_ok and deps_ok:
        print("\n✅ All checks passed! Your environment is ready to use.")
        print("You can now run the application with: python -m src.legal_doc_analysis.main")
    else:
        print("\n❌ Some checks failed. Please address the issues above before proceeding.")
    
    print("\nNote: Don't forget to add your OpenAI API key to the .env file!")
    print("You can get an API key from: https://platform.openai.com/api-keys")

if __name__ == "__main__":
    main()
