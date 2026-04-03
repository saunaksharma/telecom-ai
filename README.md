# Airtel AI Assistant

A RAG-based (Retrieval-Augmented Generation) telecom support chatbot built with Ollama, FAISS, and Streamlit. Ask natural language questions about telecom topics and get answers grounded in your own data.

---

## How It Works

```
Raw CSV → Ingest → Transform → Embed → FAISS Index → Query → LLM Answer
```

1. **Ingest** (`pipelines/ingest.py`) — cleans the raw telecom Q&A CSV (deduplication, whitespace normalization)
2. **Transform** (`pipelines/transform.py`) — formats cleaned data into a `text` column ready for embedding
3. **Embed** (`src/ingest.py`) — generates embeddings via Ollama and stores them in a FAISS index
4. **Query** (`src/query.py`) — takes a user question, retrieves the top-k similar docs from FAISS, and sends them as context to the LLM
5. **UI** (`app.py`) — Streamlit chat interface with retrieval depth control and context viewer

---

## Stack

| Component   | Tech                  |
|-------------|----------------------|
| LLM         | Ollama (`llama3:8b`) |
| Embeddings  | Ollama (`llama3:8b`) |
| Vector DB   | FAISS                |
| UI          | Streamlit            |
| Data        | Pandas               |

---

## Prerequisites

- [Ollama](https://ollama.com) installed and running locally
- `llama3:8b` model pulled: `ollama pull llama3:8b`
- Python 3.9+

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/saunaksharma/telecom-ai.git
cd telecom-ai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data Preparation

Place your raw telecom Q&A CSV at `data/raw/telecom.csv`. It must have these columns:

| Column     | Description          |
|------------|----------------------|
| `category` | Topic category       |
| `question` | The question text    |
| `answer`   | The answer text      |

Then run the pipeline in order:

```bash
# Step 1 — clean the raw data
python pipelines/ingest.py

# Step 2 — transform into embeddable format
python pipelines/transform.py

# Step 3 — generate embeddings and build FAISS index
python src/ingest.py
```

This creates `db/index.faiss` and `db/docs.pkl`.

---

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
telecom-ai/
├── app.py                  # Streamlit UI
├── pipelines/
│   ├── ingest.py           # Raw data cleaning
│   └── transform.py        # Data transformation
├── src/
│   ├── embed.py            # Ollama embedding client
│   ├── ingest.py           # FAISS index builder
│   └── query.py            # Retrieval + answer generation
├── data/
│   ├── raw/                # Drop your telecom.csv here
│   └── processed/          # Cleaned and transformed outputs
├── db/                     # FAISS index and docs store
├── logs/                   # Pipeline logs
└── requirements.txt
```

---

## Usage

- Use the **Retrieval Depth** slider in the sidebar to control how many documents are fetched per query (1–5)
- Expand **Retrieved Context** under any answer to see the exact source passages
- Use **Clear Chat** to reset the conversation
