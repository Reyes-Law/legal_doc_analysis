"""System requirement checks and dependency verification."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pkg_resources

from config import settings

# Required system commands and their test commands
REQUIRED_COMMANDS = {
    'tesseract': ['--version'],  # OCR
    'pdftoppm': ['-v'],  # Part of Poppler for PDF to image conversion
    'convert': ['--version'],  # Part of ImageMagick for image processing
}

# Minimum required Python version
MIN_PYTHON_VERSION = (3, 9)


def check_python_version() -> Tuple[bool, str]:
    """Check if the Python version meets the minimum requirement."""
    current_version = sys.version_info[:2]
    if current_version >= MIN_PYTHON_VERSION:
        return True, f"Python {'.'.join(map(str, current_version))} (>= {'.'.join(map(str, MIN_PYTHON_VERSION))} required)"
    return False, f"Python {'.'.join(map(str, current_version))} (>= {'.'.join(map(str, MIN_PYTHON_VERSION))} required)"


def check_command(command: str, test_args: List[str] = None) -> Tuple[bool, str]:
    """Check if a command is available and executable."""
    cmd_path = shutil.which(command)
    if not cmd_path:
        return False, f"{command} not found in PATH"
    
    if test_args:
        try:
            subprocess.run(
                [command] + test_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            return False, f"{command} test failed: {str(e)}"
    
    return True, f"{command} found at {cmd_path}"


def check_python_packages() -> Dict[str, Tuple[bool, str]]:
    """Check if required Python packages are installed."""
    with open(Path(__file__).parent.parent.parent.parent / 'requirements.txt') as f:
        required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    results = {}
    for pkg in required_packages:
        try:
            # Handle package specifications like 'package>=1.0.0'
            pkg_name = pkg.split('>')[0].split('=')[0].strip()
            version = pkg_resources.get_distribution(pkg_name).version
            results[pkg_name] = (True, f"{pkg_name}=={version} (installed)")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
            results[pkg_name] = (False, f"{pkg} (missing)")
    
    return results


def check_system_requirements() -> Dict[str, Dict[str, Tuple[bool, str]]]:
    """Check all system requirements and return a status report."""
    results = {
        'python': {'Python Version': check_python_version()},
        'system_commands': {},
        'python_packages': check_python_packages(),
    }
    
    # Check required system commands
    for cmd, test_args in REQUIRED_COMMANDS.items():
        results['system_commands'][cmd] = check_command(cmd, test_args)
    
    return results


def print_requirement_status(requirements: Dict[str, Dict[str, Tuple[bool, str]]]):
    """Print a formatted status of system requirements."""
    print("\n=== System Requirements Check ===\n")
    
    # Python version
    print("Python:")
    status, message = requirements['python']['Python Version']
    status_str = "[PASS]" if status else "[FAIL]"
    print(f"  {status_str} {message}")
    
    # System commands
    print("\nSystem Commands:")
    all_commands_ok = True
    for cmd, (status, message) in requirements['system_commands'].items():
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str} {cmd}: {message}")
        all_commands_ok = all_commands_ok and status
    
    # Python packages
    print("\nPython Packages:")
    all_packages_ok = True
    for pkg, (status, message) in requirements['python_packages'].items():
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"  {status_str} {message}")
        all_packages_ok = all_packages_ok and status
    
    print("\n=== Summary ===")
    if all([
        requirements['python']['Python Version'][0],
        all_commands_ok,
        all_packages_ok
    ]):
        print("\n[SUCCESS] All requirements are satisfied!")
        return True
    else:
        print("\n[WARNING] Some requirements are not met. See above for details.")
        return False


def install_system_dependencies() -> bool:
    """Install system dependencies based on the current platform."""
    system = platform.system().lower()
    
    print("Installing system dependencies...")
    
    try:
        if system == 'darwin':  # macOS
            print("\n=== macOS Detected ===")
            print("Installing dependencies using Homebrew...")
            subprocess.run([
                'brew', 'install', 'tesseract', 'tesseract-lang', 'poppler', 'imagemagick'
            ], check=True)
            print("\nDependencies installed successfully!")
            return True
            
        elif system == 'linux':
            print("\n=== Linux Detected ===")
            # Try to detect package manager
            if shutil.which('apt-get'):  # Debian/Ubuntu
                print("Installing dependencies using apt-get...")
                subprocess.run([
                    'sudo', 'apt-get', 'update'
                ], check=True)
                subprocess.run([
                    'sudo', 'apt-get', 'install', '-y',
                    'tesseract-ocr', 'tesseract-ocr-eng', 'poppler-utils', 'imagemagick'
                ], check=True)
                print("\nDependencies installed successfully!")
                return True
            else:
                print("Unsupported Linux distribution or package manager.")
                return False
                
        elif system == 'windows':
            print("\n=== Windows Detected ===")
            print("Please install the following manually:")
            print("1. Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Poppler: https://github.com/oschwartz10612/poppler-windows/")
            print("3. ImageMagick: https://imagemagick.org/script/download.php")
            print("\nMake sure to add them to your system PATH after installation.")
            return False
            
        else:
            print(f"Unsupported operating system: {system}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\nError installing dependencies: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False
