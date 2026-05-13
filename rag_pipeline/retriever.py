"""
rag_pipeline/retriever.py
==========================
Wraps the FAISS vectorstore in a LangChain retriever.
The retriever converts a natural-language query into an embedding,
performs ANN (approximate nearest-neighbour) search in the FAISS index,
and returns the top-k most semantically similar chunks.
"""

import logging
from langchain_community.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever

logger = logging.getLogger(__name__)

# Number of chunks to fetch per query.
# Higher k → more context for the LLM but also more tokens consumed.
DEFAULT_K = 4

# Search type: "similarity" (cosine / L2) or "mmr" (Maximal Marginal Relevance)
# MMR balances relevance with diversity, reducing redundant retrieved chunks.
SEARCH_TYPE = "mmr"

# MMR-specific: fetch_k candidates before re-ranking with diversity penalty
MMR_FETCH_K = 10


def build_retriever(vectorstore: FAISS) -> BaseRetriever:
    """
    Build a LangChain retriever from a FAISS vectorstore.

    Args:
        vectorstore: A populated FAISS index with embedded document chunks.

    Returns:
        BaseRetriever: Configured retriever ready for use in a chain.
    """
    logger.info(
        f"Building retriever – type={SEARCH_TYPE}, k={DEFAULT_K}"
    )

    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": DEFAULT_K,
            "fetch_k": MMR_FETCH_K,  # Only used when search_type="mmr"
        },
    )

    logger.info("Retriever ready ✓")
    return retriever
