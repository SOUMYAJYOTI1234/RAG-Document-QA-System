# 📄 RAG Document Q&A System

A production-ready **Retrieval-Augmented Generation** pipeline that lets users
upload PDF documents and ask natural-language questions about them.  
Answers are grounded in the uploaded content and include **source citations**.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Streamlit Chat UI                           │
│          (file upload · conversation · source citations)           │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  HTTP (JSON)
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                     FastAPI  Backend                                │
│                                                                    │
│  POST /upload ──► ingestor.py                                      │
│       │            ├─ PyMuPDF: extract text page-by-page           │
│       │            ├─ LangChain: RecursiveCharacterTextSplitter    │
│       │            ├─ sentence-transformers: embed chunks          │
│       │            └─ FAISS: persist index to disk                 │
│       │                                                            │
│  POST /ask ───► retriever.py ──► generator.py                      │
│       │          ├─ Load FAISS     ├─ Groq LLM (llama3-8b)        │
│       │          └─ Top-5 chunks   └─ Custom RAG prompt            │
│       │                                                            │
│       └──────► monitor.py                                          │
│                 └─ SQLite: log query, chunks, answer, latency      │
│                                                                    │
│  GET /logs ──► monitor.py ──► return recent query logs             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
rag-qa-system/
├── app/
│   ├── __init__.py        # Package marker
│   ├── ingestor.py        # PDF loading, chunking, embedding, FAISS indexing
│   ├── retriever.py       # FAISS similarity search, top-k chunk retrieval
│   ├── generator.py       # LangChain chain: retrieved context + LLM answer
│   ├── monitor.py         # SQLite logging of queries, chunks, answers
│   └── api.py             # FastAPI app with /upload and /ask endpoints
├── streamlit_app.py       # Streamlit chat UI calling the FastAPI backend
├── data/                  # Uploaded PDFs stored here
├── vector_store/          # FAISS index persisted here
├── logs/                  # SQLite DB stored here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone & install

```bash
git clone https://github.com/your-username/rag-qa-system.git
cd rag-qa-system

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Edit `.env` and paste your Groq API key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Run the FastAPI backend

```bash
uvicorn app.api:app --reload --port 8000
```

### 4. Run the Streamlit UI (new terminal)

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker

```bash
# Copy and edit your .env first
copy .env.example .env

docker-compose up --build
```

- **API** → http://localhost:8000/docs
- **UI**  → http://localhost:8501

---

## 🔑 Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to **API Keys** in the sidebar
4. Click **Create API Key**
5. Copy the key and paste it into your `.env` file

The free tier offers generous rate limits for development and demos.

---

## 💡 Example Queries

After uploading a PDF document, try questions like:

| Query | Expected Behavior |
|-------|-------------------|
| "What is the main topic of this document?" | Summarises the document's theme with page citations |
| "List the key findings mentioned in section 3." | Extracts specific details from the relevant section |
| "What methodology was used in the study?" | Pulls methodology details with source references |
| "Who are the authors?" | Extracts author info, or says "I don't know" if not in context |

---

## 🛠️ Skills Demonstrated

| Skill | Details |
|-------|---------|
| **RAG Pipeline Design** | End-to-end retrieval-augmented generation architecture |
| **LangChain** | RetrievalQA chains, custom prompt templates, document loaders |
| **FAISS Vector Search** | Local vector database with similarity search and index persistence |
| **Embedding Models** | sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings |
| **FastAPI** | RESTful API with Pydantic validation, error handling, file uploads |
| **Streamlit** | Interactive chat UI with file upload and monitoring dashboard |
| **Docker** | Multi-service containerization with docker-compose |
| **Experiment Monitoring** | SQLite-based query logging with latency tracking |
| **Prompt Engineering** | Custom grounded QA prompt with citation instructions |
| **Python Best Practices** | Type hints, docstrings, PEP 8, dotenv, error handling |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
