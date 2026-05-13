"""
rag_pipeline/text_splitter.py
==============================
Splits raw Document pages into smaller, overlapping chunks suitable
for embedding and retrieval. Overlap preserves cross-boundary context.
"""

import logging
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Chunk configuration ───────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # Max characters per chunk
CHUNK_OVERLAP = 200    # Characters shared between adjacent chunks (context bridge)
# Separators tried in order; falls back to char-level when none match
SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries semantic boundaries
    (paragraphs → sentences → words → characters) before hard-splitting.

    Args:
        documents: Raw Document list from document_loader.

    Returns:
        List[Document]: Chunked documents with inherited metadata.

    Raises:
        ValueError: If splitting yields no chunks.
    """
    logger.info(
        f"Splitting {len(documents)} pages "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        add_start_index=True,  # Adds 'start_index' to metadata for debugging
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting produced no chunks. Check your PDFs.")

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")

    # Log a sample chunk for debugging
    if chunks:
        sample = chunks[0]
        logger.debug(
            f"Sample chunk – source: {sample.metadata.get('source')}, "
            f"length: {len(sample.page_content)} chars"
        )

    return chunks
