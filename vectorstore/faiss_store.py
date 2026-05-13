"""
vectorstore/faiss_store.py
===========================
Manages FAISS vectorstore lifecycle:
  - Build from document chunks + embeddings
  - Persist to disk (for session reuse)
  - Load from disk (for warm starts)

FAISS (Facebook AI Similarity Search) provides:
  - Sub-millisecond nearest-neighbour search
  - In-memory operation (no external DB needed)
  - IVF / HNSW index types for large corpora
"""

import os
import logging
from typing import List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Default persistence directory
VECTORSTORE_PATH = Path("vectorstore/faiss_index")


def build_vectorstore(chunks: List[Document], embeddings) -> FAISS:
    """
    Build a new FAISS index from document chunks.

    Internally:
      1. Calls embeddings.embed_documents() on all chunk texts.
      2. Builds an L2-indexed flat FAISS index.
      3. Wraps it in LangChain's FAISS class for easy querying.

    Args:
        chunks:     List of Document objects (split text + metadata).
        embeddings: Any LangChain-compatible embedding model.

    Returns:
        FAISS: Populated vectorstore.
    """
    logger.info(f"Building FAISS index from {len(chunks)} chunks…")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    logger.info("FAISS index built ✓")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Path = VECTORSTORE_PATH) -> None:
    """
    Persist the FAISS index to disk.

    Saves two files:
      - index.faiss  (binary index)
      - index.pkl    (docstore + metadata)

    Args:
        vectorstore: The FAISS instance to save.
        path:        Directory to save into (created if missing).
    """
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))
    logger.info(f"Vectorstore saved to '{path}' ✓")


def load_vectorstore(embeddings, path: Path = VECTORSTORE_PATH) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        embeddings: Must match the model used when saving (same vector dim).
        path:       Directory containing index.faiss + index.pkl.

    Returns:
        FAISS: Restored vectorstore.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No saved vectorstore found at '{path}'. "
            "Upload and process documents first."
        )

    logger.info(f"Loading vectorstore from '{path}'…")
    vectorstore = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,  # Required by LangChain ≥ 0.1.17
    )
    logger.info("Vectorstore loaded ✓")
    return vectorstore
