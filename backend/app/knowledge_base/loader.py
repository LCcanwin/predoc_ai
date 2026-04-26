"""Knowledge base doc file loader using Unstructured or python-docx as fallback."""

import os
from pathlib import Path
from typing import Iterator
from docx import Document as DocxDocument


def load_docx(file_path: str) -> Iterator[str]:
    """Load text from .docx files using python-docx."""
    doc = DocxDocument(file_path)
    for para in doc.paragraphs:
        if para.text.strip():
            yield para.text.strip()


def load_txt(file_path: str) -> Iterator[str]:
    """Load text from .txt files."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.strip()


def load_documents(directory: str) -> Iterator[tuple[str, str]]:
    """
    Load all knowledge base documents from a directory.

    Yields:
        tuples of (filename, text_content)
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for file_path in dir_path.iterdir():
        if file_path.suffix.lower() == ".docx":
            for text in load_docx(str(file_path)):
                yield (file_path.name, text)
        elif file_path.suffix.lower() == ".txt":
            for text in load_txt(str(file_path)):
                yield (file_path.name, text)
        elif file_path.suffix.lower() == ".doc":
            # Fallback for older .doc format - try as text first
            try:
                for text in load_txt(str(file_path)):
                    yield (file_path.name, text)
            except Exception:
                # Skip unsupported .doc files
                continue
