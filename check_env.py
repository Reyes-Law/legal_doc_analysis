import sys
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError

def check_python_version():
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")

def check_package(package_name):
    try:
        ver = version(package_name)
        print(f"✅ {package_name}: {ver}")
        return True
    except PackageNotFoundError:
        print(f"❌ {package_name}: Not installed")
        return False

def check_system_dependency(cmd, name):
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {name} found at: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {name} not found")
            return False
    except Exception as e:
        print(f"❌ Error checking {name}: {str(e)}")
        return False

def main():
    print("\n=== Environment Check ===")
    check_python_version()
    
    print("\n=== System Dependencies ===")
    check_system_dependency("tesseract", "Tesseract OCR")
    check_system_dependency("pdftoppm", "Poppler (pdftoppm)")
    
    print("\n=== Python Packages ===")
    packages = [
        "fastapi", "uvicorn", "python-multipart", "python-dotenv", "watchdog",
        "pytesseract", "pdf2image", "PyPDF2", "python-pptx", "python-docx",
        "extract-msg", "spacy", "langchain", "openai", "sentence-transformers",
        "sqlalchemy"
    ]
    
    all_installed = True
    for pkg in packages:
        if not check_package(pkg):
            all_installed = False
    
    if all_installed:
        print("\n✅ All required packages are installed!")
    else:
        print("\n❌ Some packages are missing. Please install them using 'pip install -r requirements.txt'")

if __name__ == "__main__":
    main()
