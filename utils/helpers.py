"""
utils/helpers.py
=================
Shared utility functions used across the Streamlit app.
"""

import time
import streamlit as st
from langchain.schema import Document


def format_sources(index: int, doc: Document) -> str:
    """
    Format a retrieved Document into a readable markdown source card.

    Args:
        index: Display number (1-based).
        doc:   LangChain Document with page_content and metadata.

    Returns:
        Markdown-formatted string for display in Streamlit.
    """
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    source   = meta.get("source", "Unknown source")
    page     = meta.get("page", "?")
    snippet  = doc.page_content[:300] if hasattr(doc, "page_content") else str(doc)

    return (
        f"**📄 Source {index}:** `{source}` — Page {page}\n\n"
        f"> {snippet}{'…' if len(doc.page_content) > 300 else ''}"
    )


def stream_text(placeholder, text: str, delay: float = 0.012) -> None:
    """
    Simulate a streaming / typewriter effect in Streamlit.

    Renders the text word by word into a st.empty() placeholder.
    This mimics LLM token streaming without requiring actual streaming mode.

    Args:
        placeholder: A st.empty() placeholder element.
        text:        Full response text to render.
        delay:       Seconds between each word reveal.
    """
    words = text.split(" ")
    displayed = ""
    for word in words:
        displayed += word + " "
        placeholder.markdown(displayed + "▌")  # blinking cursor
        time.sleep(delay)
    placeholder.markdown(text)  # Final render without cursor


def get_welcome_message(documents_loaded: bool) -> str:
    """
    Return a context-aware welcome message based on the app state.

    Args:
        documents_loaded: Whether the knowledge base is ready.

    Returns:
        Markdown-formatted welcome string.
    """
    if documents_loaded:
        return (
            "👋 **Welcome back!** Your knowledge base is ready.\n\n"
            "Ask me anything about the documents you've uploaded — "
            "I'll find the most relevant information and provide accurate answers.\n\n"
            "*Try asking: 'What is the main topic of these documents?' or "
            "'Summarize the key points.'*"
        )
    return (
        "👋 **Welcome to the AI Support Assistant!**\n\n"
        "I'm a RAG-powered chatbot that answers questions from your documents.\n\n"
        "**To get started:**\n"
        "1. 📂 Upload one or more PDF files in the sidebar\n"
        "2. ⚡ Click **Process Documents** to build the knowledge base\n"
        "3. 💬 Ask me anything!\n\n"
        "*I'll reference the exact sections of your documents to support every answer.*"
    )
