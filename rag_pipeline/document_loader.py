"""
rag_pipeline/document_loader.py
================================
Handles loading PDF files from Streamlit UploadedFile objects.
Uses LangChain's PyPDFLoader under the hood.
"""

import tempfile
import os
import logging
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


def load_pdfs(uploaded_files) -> List[Document]:
    """
    Load and parse one or more PDF files uploaded via Streamlit.

    Strategy:
      1. Write each UploadedFile to a temp file on disk.
      2. Feed the temp path to LangChain's PyPDFLoader.
      3. Collect all pages as LangChain Document objects.
      4. Annotate each Document with its source filename.

    Args:
        uploaded_files: List of streamlit UploadedFile objects.

    Returns:
        List[Document]: All pages from all PDFs as LangChain Documents.

    Raises:
        ValueError: If no documents could be extracted.
    """
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        logger.info(f"Loading PDF: {uploaded_file.name}")

        # Write the in-memory file to a temporary path so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Tag each page with the original filename for source attribution
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.metadata["file_path"] = uploaded_file.name

            all_docs.extend(pages)
            logger.info(f"  → Extracted {len(pages)} pages from '{uploaded_file.name}'")

        except Exception as e:
            logger.error(f"Failed to load '{uploaded_file.name}': {e}")
            raise

        finally:
            # Always clean up the temp file
            os.unlink(tmp_path)

    if not all_docs:
        raise ValueError("No text could be extracted from the provided PDFs.")

    logger.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs
