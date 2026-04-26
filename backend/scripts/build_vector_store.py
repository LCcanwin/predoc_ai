#!/usr/bin/env python3
"""Build the FAISS vector store from project knowledge-base documents."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DATA_PATH, VECTOR_STORE_PATH
from app.knowledge_base.vector_store import build_vector_store_from_directory


def main():
    print(f"Building vector store from: {DATA_PATH}")
    vector_store = build_vector_store_from_directory(DATA_PATH, VECTOR_STORE_PATH)
    print(f"Vector store saved to: {VECTOR_STORE_PATH}")

    # Verify
    docs = vector_store.similarity_search("感冒", k=2)
    print(f"Search test for '感冒': found {len(docs)} results")
    if docs:
        print(f"First result (truncated): {docs[0].page_content[:100]}...")


if __name__ == "__main__":
    main()
