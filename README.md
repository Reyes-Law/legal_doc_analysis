# Legal Document Analysis Pipeline

An automated document processing system for legal case management that ingests, processes, and analyzes legal documents to generate medical chronologies and answer case-specific queries.

## Features

- **Document Ingestion**: Automatically process zip files containing case documents
- **Multi-format Support**: Handles PDFs, Word docs, images, emails, and more
- **OCR Capabilities**: Extracts text from scanned documents and images
- **Document Classification**: Categorizes documents by type (medical records, correspondence, etc.)
- **AI-Powered Analysis**: Generates medical chronologies and answers case-related questions
- **Local-First Architecture**: Runs entirely on your local machine

## Prerequisites

### System Requirements

- Python 3.9+
- Tesseract OCR (for OCR functionality)
- Poppler (for PDF processing)
- Development tools (compiler, make, etc.)

### Python Dependencies

All Python dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

### Installing System Dependencies

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install tesseract tesseract-lang poppler
```

#### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng poppler-utils
```

#### Windows

1. Download and install Tesseract OCR from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download and install Poppler from [poppler-for-windows](https://github.com/oschwartz10612/poppler-windows/)
3. Add both to your system PATH

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd legal-doc-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (copy .env.example to .env and update values):
   ```bash
   cp .env.example .env
   ```

## Usage

1. Place your case zip files in the `data/input` directory
2. Run the processing pipeline:
   ```bash
   python -m src.legal_doc_analysis.main
   ```
3. Access the web interface at `http://localhost:8000`

## Development

### Setting up for Development

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

2. Run tests:
   ```bash
   pytest
   ```

### Project Structure

- `src/legal_doc_analysis/` - Main package
  - `document_ingestion/` - File watching and extraction
  - `processing/` - Document processing pipeline
  - `analysis/` - AI and analysis components
  - `models/` - Database models
  - `api/` - Web interface/API
- `config/` - Configuration files
- `data/` - Data directories (input, processed, database)
- `tests/` - Test files

## License

This project is proprietary and confidential.

## Support

For support, please contact your system administrator.
