# Enterprise Document Q&A Bot (RAG Pipeline)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** pipeline built in Python. This system ingests corporate documents (PDFs like employee handbooks, SOPs, compliance manuals), parses and chunks the text with boundary preservation, maps passages into dense vector embeddings using open-source Hugging Face models, indexes them in a persistent vector database, and performs cosine-similarity semantic search with strict anti-hallucination guardrails.

---

## 🏗️ Architecture & Data Flow

```text
[Offline Indexing Phase]
  PDF Document 
      ↓
  Text Extraction (pypdf / PyMuPDF)
      ↓
  Overlapping Sliding-Window Chunking (500 chars, 50 overlap)
      ↓
  Hugging Face Embedding Model (sentence-transformers/all-MiniLM-L6-v2)
      ↓
  Local Vector Database (384-dimensional Normalized Vectors + Text + Metadata)

[Online Query Phase]
  User Question
      ↓
  Query Vectorization (384-d vector)
      ↓
  Cosine Similarity Search (Thresholded: ≥ 0.25)
      ↓
  Context-Augmented Prompt Assembly (Numbered snippet citations)
      ↓
  LLM Generation (OpenAI / Local Grounded Engine)
      ↓
  Factual Answer with Source Citations
```

---

## 🚀 Key Features

- **Document Parsing:** High-fidelity text extraction supporting both `pypdf` and `PyMuPDF` (`fitz`).
- **Semantic Chunking:** Overlapping sliding window (500 characters, 50-character overlap) to prevent critical sentences from breaking across chunk boundaries.
- **Open-Source Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face `transformers` & PyTorch with Mean Pooling and L2 Normalization.
- **Dual Vector Database:** Production-ready persistence supporting both ChromaDB and an integrated zero-dependency vector engine with exact Cosine Similarity.
- **Anti-Hallucination Guardrails:**
  - **Similarity Thresholding:** Out-of-domain questions with cosine similarity below `0.25` are flagged and rejected immediately.
  - **Negative Grounding Constraints:** Prompts strictly instruct the model to refuse to answer if the context does not contain the facts.
- **Flexible Generation:** Supports OpenAI API (`gpt-3.5-turbo` / `gpt-4o`) when an API key is present, with an automated fallback to a local grounded extraction engine so it runs completely free.

---

## 📁 Repository Structure

```text
├── rag_pipeline.py         # Main RAG pipeline (Extraction -> Chunking -> Embedding -> Search -> QA)
├── create_sample_pdf.py    # Generates a sample corporate employee handbook PDF
├── smoke_test.py           # Comprehensive 8-point automated test suite
├── requirements.txt        # Dependency definitions
├── run.bat                 # One-click Windows launch script
└── .gitignore              # Ignores virtual environments, caches, and DB files
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Rehan-ctr/Enterprise-Document-Q-A-Bot-.git
cd Enterprise-Document-Q-A-Bot-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate sample corporate document
```bash
python create_sample_pdf.py
```

### 4. Run the automated smoke test suite
```bash
python smoke_test.py
```

### 5. Launch the interactive Q&A Bot
```bash
python rag_pipeline.py
```
*(Optional: Set `OPENAI_API_KEY="sk-..."` in your environment to enable OpenAI generation. If not set, the built-in local engine will generate answers).*

---

## 🧪 Validation & Test Results

The pipeline includes an end-to-end verification script testing all components:

```text
=================================================================
  RAG PIPELINE END-TO-END SMOKE TEST
=================================================================
  [PASS]  PDF Extraction
  [PASS]  Text Chunking
  [PASS]  Hugging Face Embedding Model Load
  [PASS]  Vector Database Setup
  [PASS]  Passage Ingestion & Upsert
  [PASS]  Semantic Similarity Retrieval
  [PASS]  Prompt Construction
  [PASS]  Grounded Answer Generation
=================================================================
  ALL 8 TESTS PASSED SUCCESSFULLY! (100% HEALTHY)
```

---

## 🛡️ License
MIT License. Free for enterprise and personal educational use.
