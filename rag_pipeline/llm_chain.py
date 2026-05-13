"""
rag_pipeline/llm_chain.py
==========================
Builds the full RAG pipeline using LangGraph + LangChain.

Architecture:
  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────┐
  │  User Query  │───▶│  Retriever  │───▶│  Prompt Tmpl │───▶│   LLM    │
  │ + Chat Hist  │    │ (FAISS MMR) │    │ (Context Inj)│    │ Response │
  └──────────────┘    └─────────────┘    └──────────────┘    └──────────┘

LangGraph is used to build a stateful, multi-node workflow graph that can
be extended with routing, reflection, or tool-use nodes later.

Supported LLM backends (priority order):
  1. Google Gemini  (GOOGLE_API_KEY)
  2. OpenAI GPT-4o  (OPENAI_API_KEY)
  3. OpenAI GPT-3.5 (fallback)
"""

import os
import logging
from typing import List, Tuple, TypedDict, Annotated

from langchain.schema import Document, BaseRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LangGraph imports
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, professional AI customer support assistant.
Your job is to answer questions accurately based ONLY on the provided context documents.

Guidelines:
- Answer clearly and concisely using the context provided.
- If the answer is not in the context, say "I don't have information about that in the provided documents."
- Never fabricate facts. Stick to the source material.
- If asked about topics outside the documents, politely redirect.
- Use bullet points or numbered lists when explaining multi-step processes.
- Be empathetic and professional in tone.

Context:
{context}
"""

# ── LangGraph State Schema ────────────────────────────────────────────────────
class RAGState(TypedDict):
    """State object passed between nodes in the LangGraph workflow."""
    question: str
    chat_history: List[Tuple[str, str]]
    context: List[Document]
    answer: str
    source_documents: List[Document]


# ── LLM Loader ────────────────────────────────────────────────────────────────
def _load_llm():
    """
    Load the best available LLM from environment variables.
    Tries Gemini → OpenAI in order.
    """
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        logger.info("Using Google Gemini as LLM backend")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_key,
                temperature=0.3,
                convert_system_message_to_human=True,
            )
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}. Falling back to OpenAI.")

    if openai_key:
        logger.info("Using OpenAI as LLM backend")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_key,
            temperature=0.3,
            streaming=False,
        )

    raise RuntimeError(
        "No LLM API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file."
    )


# ── Format Helpers ────────────────────────────────────────────────────────────
def _format_docs(docs: List[Document]) -> str:
    """Join retrieved documents into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def _format_chat_history(history: List[Tuple[str, str]]) -> List:
    """Convert (human, ai) tuples into LangChain message objects."""
    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


# ── LangGraph Nodes ───────────────────────────────────────────────────────────
def make_retrieve_node(retriever: BaseRetriever):
    """Node 1: Retrieve relevant chunks from the vectorstore."""
    def retrieve(state: RAGState) -> RAGState:
        logger.info(f"[Node: retrieve] Query: {state['question'][:60]}…")
        docs = retriever.invoke(state["question"])
        logger.info(f"[Node: retrieve] Got {len(docs)} chunks")
        return {**state, "context": docs, "source_documents": docs}
    return retrieve


def make_generate_node(llm, prompt):
    """Node 2: Generate an answer given context + chat history."""
    chain = prompt | llm | StrOutputParser()

    def generate(state: RAGState) -> RAGState:
        logger.info("[Node: generate] Building response…")
        context_str = _format_docs(state["context"])
        history_msgs = _format_chat_history(state.get("chat_history", []))

        answer = chain.invoke(
            {
                "context": context_str,
                "chat_history": history_msgs,
                "question": state["question"],
            }
        )
        logger.info("[Node: generate] Response ready")
        return {**state, "answer": answer}

    return generate


# ── Chain Builder ─────────────────────────────────────────────────────────────
def build_rag_chain(retriever: BaseRetriever):
    """
    Compile a LangGraph workflow with two nodes:
      retrieve → generate → END

    Returns a compiled graph that accepts {"question": ..., "chat_history": ...}
    and returns {"answer": ..., "source_documents": ...}.
    """
    llm = _load_llm()

    # Build the prompt template with conversation memory slot
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Define the graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("generate", make_generate_node(llm, prompt))

    # Define edges: retrieve → generate → END
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Compile (validates graph, builds execution plan)
    compiled = graph.compile()
    logger.info("LangGraph RAG chain compiled ✓")
    return compiled
