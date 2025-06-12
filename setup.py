from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="legal_doc_analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Legal Document Analysis Pipeline for Case Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/legal-doc-analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies will be read from requirements.txt
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
        "Topic :: Office/Business",
    ],
    entry_points={
        "console_scripts": [
            "legal-doc-analyze=legal_doc_analysis.cli:main",
        ],
    },
)
