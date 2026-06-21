# Document Q&A Bot — RAG (Retrieval-Augmented Generation)

A locally-running Q&A system that answers questions strictly grounded in your own
PDF and DOCX documents, with inline source citations and zero hallucinated facts.

Built as part of an AI Engineering internship project on Retrieval-Augmented Generation.

---
## live
https://document-q-a-bot-b4xqrjve5cflguuxzeddcb.streamlit.app/


## How It Works

```
 Your Documents (PDF/DOCX)
          │
          ▼
   Extract text + page metadata
          │
          ▼
   Chunk into overlapping segments
          │
          ▼
   Embed chunks (Gemini text-embedding-004)
          │
          ▼
   Store in ChromaDB (persisted to disk)

 ── at query time ──

 User question
      │
      ▼
 Embed question (same model)
      │
      ▼
 Retrieve top-k most similar chunks
      │
      ▼
 Build grounded prompt with citations
      │
      ▼
 Gemini generates answer using ONLY retrieved context
      │
      ▼
 Answer + citations shown in Streamlit UI
```

The system never lets the LLM answer from its own general knowledge — it is
instructed to answer *only* from the retrieved document chunks, and to say so
explicitly when the answer isn't in the documents.

---



## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv


# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Get a free Gemini API key at https://aistudio.google.com/app/apikey, then:

```bash
cp .env.example .env
```

Edit `.env` and paste your key:

```
GEMINI_API_KEY=your_actual_key_here
```

### 4. Add your documents

Sample placeholder documents are already included in `data/` so you can test
the pipeline immediately. Replace them with your own `.pdf` / `.docx` files
whenever you're ready — just drop files into `data/` and re-run ingestion.

---

## Usage

### Step 1 — Ingest your documents (run once, or whenever documents change)

```bash
python src/ingest.py
```

This will:
- Scan `data/` for `.pdf` and `.docx` files
- Extract text with page-level metadata
- Chunk text into overlapping ~1000-character segments
- Embed each chunk with Gemini's `text-embedding-004`
- Persist everything to `db/` (a local ChromaDB store)



### Step 2 — Launch the Q&A interface

```bash
streamlit run src/main.py
```

This opens a browser tab with a chat interface. Ask questions like:

- *"What was the net revenue for FY2026?"*
- *"Who is the CEO of Northwind Retail Group?"*
- *"What moisture level produced the highest fungal colonization rate?"*
- *"What is the company's revenue from 1999?"* (not in the docs — the bot will correctly say it can't find the answer, instead of guessing)

Each answer includes citation pills (filename + page) and an expandable
"View retrieved source chunks" panel so you can verify exactly what context
the model used.

### Quick CLI test without the UI

```bash
python src/query.py "What was the net revenue for FY2026?"
```

---

## Configuration

All tunable parameters live in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K` | 4 | Number of chunks retrieved per query |
| `DISTANCE_THRESHOLD` | 1.0 | Max cosine distance for a chunk to count as relevant |
| `GENERATION_MODEL` | `gemini-2.5-flash-preview-09-2025` | LLM used for answer generation |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |

The retrieval count (`k`) can also be adjusted live from a slider in the
Streamlit sidebar.

---

## Design Notes

- **Why ChromaDB?** It's a lightweight, disk-persistent vector store that
  requires no separate server process — ideal for a self-contained local project.
- **Why recursive/overlapping chunking?** Splitting purely by character count
  risks cutting a sentence (and its meaning) in half at a chunk boundary. The
  200-character overlap ensures context on either side of a cut is preserved
  in both neighboring chunks.
- **Why a distance threshold?** Without it, the bot would always return its
  `top_k` chunks even if none of them are actually related to the question,
  leading to confident-sounding but irrelevant or hallucinated answers.
- **Idempotent ingestion:** Re-running `ingest.py` wipes and rebuilds the
  `db/` collection from scratch rather than appending, so stale chunks from
  deleted/renamed source files never linger.
- **DOCX vs PDF pagination:** PDFs have native page boundaries used directly
  for citations. DOCX files don't have a fixed page concept in the file format
  itself, so the whole document is treated as one citable unit (page 1); for
  longer DOCX files this gets split into multiple overlapping chunks by the
  same chunking step, just without per-page granularity.

---

## Known Limitations

- Citation accuracy depends entirely on chunk-level metadata; very large DOCX
  files will cite "Page 1" for every chunk since DOCX doesn't track pages.
- The distance threshold and `top_k` are heuristics — tune them in
  `config.py` based on your own document set and embedding distribution.
- No OCR: scanned/image-only PDFs without embedded text will extract no
  content. Run them through an OCR tool first if needed.

---