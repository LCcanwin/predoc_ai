"""Semantic chunker for knowledge base documents.

Splits documents by symptom-diagnosis-treatment patterns (病症-辨证-治则).
"""

import re
from typing import Iterator
from dataclasses import dataclass


@dataclass
class Chunk:
    """A semantic chunk from a document."""
    content: str
    source: str
    chunk_index: int


# Patterns that indicate semantic boundaries
BOUNDARY_PATTERNS = [
    r"(?:^|\n)#{1,3}\s*",  # Markdown headers
    r"(?:^|\n)(?:病症|证候|辨证|治则|治疗|诊断|临床表现)",  # TCM section keywords
    r"(?:^|\n)\d+[.、]\s*",  # Numbered lists
    r"(?:^|\n)[一二三四五六七八九十]+[、.]\s*",  # Chinese numbered lists
    r"\n\n+",  # Double newlines
]


def chunk_by_semantic_splits(text: str, source: str, min_chunk_size: int = 100) -> Iterator[Chunk]:
    """
    Split text into semantic chunks based on structural patterns.

    Args:
        text: Document text to chunk
        source: Source filename
        min_chunk_size: Minimum chunk size in characters

    Yields:
        Chunk objects with semantic content
    """
    # Try to split by major patterns first
    split_pattern = "|".join(BOUNDARY_PATTERNS)
    parts = re.split(split_pattern, text)

    chunks = []
    current_chunk = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        current_chunk.append(part)

        if len(" ".join(current_chunk)) >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Don't forget the remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # If no good splits happened, chunk by sentence count
    if len(chunks) <= 1 and len(text) > min_chunk_size:
        chunks = chunk_by_sentences(text, min_chunk_size)

    for idx, content in enumerate(chunks):
        yield Chunk(content=content, source=source, chunk_index=idx)


def chunk_by_sentences(text: str, min_chunk_size: int = 100) -> list[str]:
    """
    Fallback chunking by sentences when semantic splitting fails.
    """
    # Simple sentence splitting for Chinese text
    sentence_endings = r"[。！？\n]+"
    sentences = re.split(sentence_endings, text)

    chunks = []
    current = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        current.append(sentence)

        if len(" ".join(current)) >= min_chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks
