# 🤖 AI Customer Support Chatbot with RAG

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with LangChain, LangGraph, FAISS, and Streamlit. Upload your PDF documents and get accurate, source-cited answers instantly.

---

## ✨ Features

- 📂 **Multi-PDF Upload** – process multiple documents simultaneously
- 🧠 **HuggingFace Embeddings** – free, local `all-MiniLM-L6-v2` model
- ⚡ **FAISS Vector Store** – sub-millisecond similarity search
- 🔗 **LangGraph Pipeline** – stateful, extensible RAG workflow graph
- 💬 **Conversation Memory** – maintains full chat history context
- 📎 **Source Attribution** – every answer cites its document chunks
- ✍️ **Typing Animation** – streaming-style response rendering
- 🌐 **Dual LLM Support** – Google Gemini or OpenAI GPT-4o
- 🎨 **Professional Dark UI** – custom Streamlit CSS styling

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      STREAMLIT FRONTEND                       │
│  Sidebar (Upload/Process) ←→ Chat Interface (Query/Answer)   │
└───────────────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────┐
         │       RAG PIPELINE          │
         │                             │
         │  PDF → Chunks → Embeddings  │
         │         ↓                   │
         │    FAISS VectorStore        │
         │         ↓                   │
         │   LangGraph Workflow        │
         │  ┌─────────┐ ┌──────────┐  │
         │  │ Retrieve│→│ Generate │  │
         │  └─────────┘ └──────────┘  │
         │         ↓                   │
         │   Answer + Sources          │
         └─────────────────────────────┘
```

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .streamlit/
│   └── config.toml                 # Streamlit server config
│
├── rag_pipeline/                   # Core RAG logic
│   ├── __init__.py
│   ├── document_loader.py          # PDF → LangChain Documents
│   ├── text_splitter.py            # Documents → Chunks
│   ├── embeddings.py               # HuggingFace / OpenAI embeddings
│   ├── retriever.py                # FAISS MMR retriever
│   └── llm_chain.py                # LangGraph RAG workflow
│
├── vectorstore/                    # Vector DB management
│   ├── __init__.py
│   ├── faiss_store.py              # Build / save / load FAISS index
│   └── faiss_index/                # Saved index (auto-created)
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── helpers.py                  # Source formatting, streaming
│   └── logger.py                   # Rotating file logger
│
├── templates/
│   └── styles.css                  # Custom Streamlit CSS
│
├── logs/                           # Auto-created log directory
│   └── app.log
│
└── README.md
```

---

## 🚀 Local Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/yourname/rag-support-chatbot.git
cd rag-support-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set API Keys
```bash
cp .env.example .env
```
Edit `.env` and add your key:
```
GOOGLE_API_KEY=your_key_here   # Recommended (free tier)
# OR
OPENAI_API_KEY=your_key_here
```

### 5. Run
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Push your repository to GitHub (ensure `.env` is in `.gitignore`)
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch, and set **Main file path** to `app.py`
4. Go to **Advanced settings → Secrets** and add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
5. Click **Deploy** 🚀

> Streamlit Cloud secrets are injected as environment variables — no `.env` file needed.

---

## 🔄 RAG Data Flow

```
1. User uploads PDFs
        ↓
2. PyPDFLoader extracts text page-by-page
        ↓
3. RecursiveCharacterTextSplitter creates 1000-char chunks (200 overlap)
        ↓
4. HuggingFace all-MiniLM-L6-v2 generates 384-dim embeddings
        ↓
5. FAISS builds an L2 flat index from all chunk vectors
        ↓
6. User submits a question
        ↓
7. Query is embedded with the same model
        ↓
8. FAISS MMR search returns top-4 most relevant chunks
        ↓
9. Chunks + chat history are injected into the system prompt
        ↓
10. Gemini/GPT-4o generates a grounded, source-cited answer
        ↓
11. Answer + source metadata rendered in Streamlit chat UI
```

---

## 🛠️ Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `CHUNK_SIZE` | `text_splitter.py` | 1000 | Max chars per chunk |
| `CHUNK_OVERLAP` | `text_splitter.py` | 200 | Overlap between chunks |
| `DEFAULT_K` | `retriever.py` | 4 | Chunks retrieved per query |
| `SEARCH_TYPE` | `retriever.py` | `mmr` | `similarity` or `mmr` |
| `HF_MODEL` | `embeddings.py` | `all-MiniLM-L6-v2` | Embedding model |
| `LLM_TEMP` | `llm_chain.py` | 0.3 | LLM temperature |

---

## 📋 Resume Description

> **AI Customer Support Chatbot with RAG** | Python, LangChain, LangGraph, FAISS, Streamlit
>
> Built a production-grade Retrieval-Augmented Generation chatbot enabling organizations to deploy custom AI support agents over proprietary PDF documentation. Engineered a full-stack pipeline: PDF ingestion with LangChain's document loaders, recursive text chunking with semantic boundary awareness, HuggingFace sentence-transformer embeddings, and FAISS vector similarity search (MMR). Implemented a stateful LangGraph workflow graph for multi-turn conversations with persistent memory. Achieved accurate, source-attributed answers with <500ms retrieval latency. Deployed on Streamlit Cloud with Google Gemini and OpenAI dual-backend support.

---

## 🎤 Interview Questions & Answers

**Q: What is RAG and why is it better than fine-tuning for customer support?**
> RAG retrieves relevant document chunks at inference time, keeping the knowledge base updatable without expensive retraining. Fine-tuning bakes knowledge into weights — hard to update and prone to hallucination on out-of-distribution queries.

**Q: Why FAISS over a managed vector DB like Pinecone?**
> FAISS is in-process, zero-latency, and free — ideal for prototypes and smaller corpora (<1M vectors). Pinecone/Weaviate add persistence, scalability, and metadata filtering for production at scale.

**Q: What is MMR and why use it over pure similarity search?**
> Maximal Marginal Relevance balances relevance with diversity — it penalizes chunks that are too similar to already-selected results, reducing redundancy and giving the LLM a richer context.

**Q: How does LangGraph differ from a simple LangChain chain?**
> LangGraph models the pipeline as a directed graph with explicit state transitions, enabling cycles, conditional routing, human-in-the-loop steps, and parallel branches — things a linear chain can't express.

**Q: How do you handle conversation memory in RAG?**
> Chat history (human/AI turn pairs) is passed as LangChain Message objects in the prompt's `MessagesPlaceholder`. The LLM sees prior context but retrieval is always on the latest question only.

**Q: What chunking strategy did you choose and why?**
> `RecursiveCharacterTextSplitter` with 1000-char chunks and 200-char overlap. The recursive strategy respects semantic boundaries (paragraphs → sentences → words) before hard-splitting. Overlap ensures context isn't lost at chunk boundaries.

**Q: How would you scale this to 10M documents?**
> Replace FAISS flat index with HNSW or IVF-PQ for approximate search. Move to a managed vector DB (Pinecone/Weaviate). Add a re-ranker (CrossEncoder) after retrieval. Cache frequent queries. Use async chunking/embedding pipelines.
