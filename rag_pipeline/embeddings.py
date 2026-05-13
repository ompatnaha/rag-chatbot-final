"""
rag_pipeline/embeddings.py
===========================
Provides the embedding model used to convert text chunks into dense
vectors for similarity search.

Priority:
  1. HuggingFace sentence-transformers (free, runs locally)
  2. OpenAI embeddings (if OPENAI_API_KEY is set and hf fails)

The default model 'all-MiniLM-L6-v2' produces 384-dim vectors and is
fast enough for real-time RAG while being highly accurate for semantic search.
"""

import os
import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Best balance of speed ↔ accuracy for customer-support use cases
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Encode settings: normalize_embeddings=True → cosine similarity works correctly
ENCODE_KWARGS = {"normalize_embeddings": True}
MODEL_KWARGS   = {"device": "cpu"}   # Switch to "cuda" if a GPU is available


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached embedding model instance.

    @lru_cache(1) ensures the model is loaded only once per session,
    avoiding repeated ~300 MB downloads during development.

    Returns:
        HuggingFaceEmbeddings (or OpenAIEmbeddings as fallback)

    Raises:
        RuntimeError: If neither embedding backend can be initialized.
    """
    logger.info(f"Loading HuggingFace embeddings: {DEFAULT_HF_MODEL}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=DEFAULT_HF_MODEL,
            model_kwargs=MODEL_KWARGS,
            encode_kwargs=ENCODE_KWARGS,
        )
        # Smoke-test: embed a tiny string to confirm the model loaded
        _ = embeddings.embed_query("test")
        logger.info("HuggingFace embeddings ready ✓")
        return embeddings

    except Exception as hf_err:
        logger.warning(f"HuggingFace embeddings failed: {hf_err}. Trying OpenAI…")

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError(
                "HuggingFace embeddings failed and OPENAI_API_KEY is not set. "
                "Cannot initialise embeddings."
            ) from hf_err

        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_key,
            )
            logger.info("OpenAI embeddings ready ✓ (fallback)")
            return embeddings
        except Exception as oai_err:
            raise RuntimeError(f"Both embedding backends failed: {oai_err}") from oai_err
