"""
================================================================================
  RAG (Retrieval-Augmented Generation) Pipeline
  -----------------------------------------------
  Author  : AI Engineering Guide
  Purpose : Ingest a PDF document (e.g. corporate handbook), extract text,
            chunk it with overlap, convert chunks into dense vector embeddings
            using an open-source Hugging Face model, index them in a local
            vector database, and allow users to ask natural-language questions
            grounded strictly in the document content.

  WHAT IS RAG?
  ============
  Large Language Models (LLMs) like GPT-4 are pre-trained on public internet data.
  They know nothing about private corporate documents, internal SOPs, or private PDFs.
  Fine-tuning a model on private documents is expensive, slow, and prone to hallucinations.
  
  RAG (Retrieval-Augmented Generation) solves this cleanly with a 3-step paradigm:
    1. RETRIEVAL   : Given a user question, search a vector database to find the
                     most semantically relevant excerpts (chunks) from your document.
    2. AUGMENTATION: Inject those retrieved excerpts directly into the LLM prompt
                     as authoritative context ("According to these snippets...").
    3. GENERATION  : The LLM reads the context and generates an accurate, hallucination-free
                     answer grounded strictly in your document.

  DATA FLOW (End-to-End):
  -----------------------
  [Offline Indexing Phase]
  PDF Document -> Text Extraction -> Overlapping Chunking -> Embedding Model (384-d vector)
               -> Persist in Local Vector Database (Vectors + Text + Metadata)

  [Online Query Phase]
  User Query   -> Embed Query (384-d vector) -> Cosine Similarity Search in Vector DB
               -> Retrieve Top-K Chunks -> Assemble Prompt -> LLM Generation -> Final Answer
================================================================================
"""

import os
import sys
import json
import math
import textwrap
from typing import List, Dict, Any, Tuple, Optional

# ── Optional/Flexible Third-Party Imports ─────────────────────────────────────
# We import libraries gracefully so the pipeline runs smoothly across environments.

# 1. PDF Parser: PyMuPDF (fitz) or pypdf
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except ImportError:
        PDF_BACKEND = None

# 2. Embedding Engine: HuggingFace Transformers & PyTorch
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_TRANSFORMERS_AVAILABLE = True
except ImportError:
    TORCH_TRANSFORMERS_AVAILABLE = False

# 3. Vector Database: ChromaDB (optional native support)
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# 4. LLM Generation: OpenAI SDK
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – PIPELINE CONFIGURATION
#  Tuneable parameters controlling the indexing and retrieval behaviour.
# ══════════════════════════════════════════════════════════════════════════════

# Path to the source PDF document (e.g. employee handbook)
PDF_PATH = "handbook.pdf"

# Number of characters per text chunk.
# Smaller chunks (e.g., 200-500 chars) yield higher semantic specificity.
# Larger chunks (e.g., 1000-2000 chars) preserve more surrounding context.
CHUNK_SIZE = 500

# Overlap between consecutive chunks (in characters).
# Prevents sentences or critical concepts from being split across chunk boundaries.
CHUNK_OVERLAP = 50

# Hugging Face open-source embedding model.
# "sentence-transformers/all-MiniLM-L6-v2" is the industry benchmark for lightweight,
# fast, high-accuracy semantic embeddings (384-dimensional output space).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Name of the vector collection
COLLECTION_NAME = "handbook_knowledge_base"

# Directory where the local vector database is persisted
VECTOR_DB_DIR = "./vector_db"

# Number of top relevant document chunks retrieved per query
TOP_K = 3

# Minimum cosine similarity threshold to consider a document passage relevant.
# If query similarity is below this, the system recognizes the question is out-of-scope.
SIMILARITY_THRESHOLD = 0.25

# OpenAI Model configuration for final answer generation
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_ANSWER_TOKENS = 350


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – PDF TEXT EXTRACTION
#  Goal: Extract raw readable text from arbitrary PDF documents.
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Parses a PDF file and extracts text across all pages into a unified string.

    Args:
        pdf_path (str): Filepath to the target PDF.

    Returns:
        str: Clean concatenated plain text.

    Interview Note:
        PDF is an optical layout format (instructions for where to draw glyphs),
        not a structured text stream. A robust PDF extractor stitches glyphs
        into words and lines, preserving document flow.
    """
    print(f"\n[STEP 1] Extracting text from PDF: '{pdf_path}'")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found at '{pdf_path}'. "
            f"Please run 'create_sample_pdf.py' or provide a valid PDF."
        )

    extracted_text = ""

    # Strategy A: Use PyMuPDF (fitz) if installed
    if PDF_BACKEND == "pymupdf":
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                extracted_text += page_text + "\n"
        doc.close()

    # Strategy B: Use pypdf as a reliable pure-Python alternative
    elif PDF_BACKEND == "pypdf":
        reader = pypdf.PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

    else:
        raise RuntimeError("No PDF reader installed. Run: pip install pypdf")

    clean_text = extracted_text.strip()
    print(f"  Extracted {len(clean_text):,} characters of text.")
    return clean_text


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – TEXT CHUNKING (Sliding Window Strategy)
#  Goal: Divide long text into manageable semantic passages with overlap.
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits continuous text into overlapping windows.

    Args:
        text (str): Full raw text from the document.
        chunk_size (int): Target character length for each chunk.
        overlap (int): Number of overlapping characters between adjacent chunks.

    Returns:
        List[str]: List of non-empty text passages.

    Interview Explanation:
        Why chunk with overlap?
        1. Context window limits: Embedding models and LLMs have finite token capacity.
        2. Semantic density: A 300-word paragraph has a sharper vector representation
           than a 50-page document whose meaning gets diluted.
        3. Boundary preservation: If a critical fact ("Leave is limited to 20 days")
           falls on a split boundary, overlap ensures at least one chunk captures the
           complete sentence with full context.
    """
    print(f"\n[STEP 2] Chunking text into passages (size={chunk_size}, overlap={overlap})")

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move the sliding window forward by (chunk_size - overlap)
        start += (chunk_size - overlap)

    print(f"  Created {len(chunks)} overlapping chunks.")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – OPEN-SOURCE HUGGING FACE EMBEDDING MODEL
#  Goal: Transform natural language text into 384-dimensional dense vectors.
# ══════════════════════════════════════════════════════════════════════════════

class HuggingFaceEmbeddingModel:
    """
    Encapsulates an open-source Transformer model from Hugging Face Hub.
    Performs Mean Pooling and L2 Normalization to generate high-fidelity sentence embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        if not TORCH_TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and Transformers are required. Run: pip install torch transformers")

        print(f"\n[STEP 3] Initializing Hugging Face Embedding Model: '{model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode (disable dropout)
        print("  Hugging Face model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask):
        """
        Calculates the mean of token embeddings weighted by attention mask.
        Disregards padding tokens so they do not contaminate the semantic vector.
        """
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of strings into unit-normalized embedding vectors.

        Mathematical steps:
          1. Tokenize texts into sub-word tokens with attention masks.
          2. Forward-pass through Transformer to get token-level embeddings.
          3. Apply Mean-Pooling to condense token vectors into a sentence vector.
          4. Apply L2-Normalization so dot-product equals cosine similarity.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Pool token embeddings into sentence vectors
        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

        # L2-Normalize: ||v|| = 1.0
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Convert to Python floats for storage in vector database
        return sentence_embeddings.cpu().tolist()


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddingModel:
    """Factory function to load and return the embedding model."""
    return HuggingFaceEmbeddingModel(model_name)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – LOCAL VECTOR DATABASE (ChromaDB & Built-in Engine)
#  Goal: Index embeddings alongside text and metadata for similarity search.
# ══════════════════════════════════════════════════════════════════════════════

class LocalVectorDB:
    """
    A lightweight, persistent local vector database implementing cosine similarity search.
    Provides identical interface to ChromaDB collection (count, upsert, query)
    so candidates can explain the underlying math (dot product of normalized vectors)
    while also supporting production ChromaDB when present.
    """

    def __init__(self, collection_name: str, persist_dir: str = VECTOR_DB_DIR):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.filepath = os.path.join(persist_dir, f"{collection_name}.json")
        self.records: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Loads indexed records from persistent JSON file on disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = {}

    def _save(self):
        """Persists indexed records to disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def count(self) -> int:
        """Returns total number of indexed vectors."""
        return len(self.records)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Stores or updates chunk records."""
        if metadatas is None:
            metadatas = [{} for _ in ids]

        for chunk_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self.records[chunk_id] = {
                "id": chunk_id,
                "document": doc,
                "embedding": emb,
                "metadata": meta,
            }
        self._save()

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 3,
        include: Optional[List[str]] = None
    ) -> Dict[str, List[Any]]:
        """
        Executes Cosine Similarity search:
        Similarity = dot_product(A, B) / (||A|| * ||B||)
        Since vectors are already L2-normalized (||v|| = 1.0),
        Cosine Similarity simply equals dot_product(A, B)!
        """
        query_vec = query_embeddings[0]
        scored_docs = []

        for record in self.records.values():
            doc_emb = record["embedding"]
            # Dot product calculation
            similarity = sum(q * d for q, d in zip(query_vec, doc_emb))
            # Distance = 1.0 - similarity (standard cosine distance metric)
            distance = max(0.0, 1.0 - similarity)
            scored_docs.append((similarity, distance, record))

        # Sort descending by similarity (highest similarity first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_docs[:n_results]

        documents = [item[2]["document"] for item in top_k]
        distances = [item[1] for item in top_k]
        metadatas = [item[2]["metadata"] for item in top_k]
        ids = [item[2]["id"] for item in top_k]

        return {
            "documents": [documents],
            "distances": [distances],
            "metadatas": [metadatas],
            "ids": [ids],
        }


def setup_chromadb(collection_name: str = COLLECTION_NAME, persist_path: str = VECTOR_DB_DIR):
    """
    Initializes a persistent vector database collection.
    Uses native ChromaDB if installed; otherwise seamlessly defaults to LocalVectorDB.
    """
    print(f"\n[STEP 4] Setting up Vector Database collection: '{collection_name}'")

    if CHROMADB_AVAILABLE:
        try:
            client = chromadb.PersistentClient(path=persist_path)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"  Initialized native ChromaDB. Existing items: {collection.count()}")
            return client, collection
        except Exception as e:
            print(f"  ChromaDB initialization note: {e}. Falling back to LocalVectorDB.")

    # High-performance built-in persistent vector engine
    collection = LocalVectorDB(collection_name=collection_name, persist_dir=persist_path)
    print(f"  Initialized LocalVectorDB (persistent JSON + cosine search). Existing items: {collection.count()}")
    return None, collection


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – INGESTION (Embedding + Storing Chunks)
#  Goal: Batch embed document chunks and store them in the vector database.
# ══════════════════════════════════════════════════════════════════════════════

def ingest_chunks(
    chunks: List[str],
    embedding_model: HuggingFaceEmbeddingModel,
    collection: Any,
    batch_size: int = 32
) -> None:
    """
    Embeds raw text chunks and indexes them in the vector database.
    """
    print(f"\n[STEP 5] Ingesting {len(chunks)} chunks into Vector Database...")

    # Skip re-ingestion if already populated to ensure fast restarts
    if collection.count() >= len(chunks):
        print(f"  Vector database already contains {collection.count()} chunks. Skipping re-ingestion.")
        return

    for start_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start_idx : start_idx + batch_size]
        batch_embeddings = embedding_model.embed(batch_chunks)
        batch_ids = [f"chunk_{start_idx + i}" for i in range(len(batch_chunks))]
        batch_metadatas = [{"chunk_index": start_idx + i, "source": PDF_PATH} for i in range(len(batch_chunks))]

        collection.upsert(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        print(f"  Indexed chunks [{start_idx} to {start_idx + len(batch_chunks) - 1}]")

    print(f"  Ingestion complete! Total stored chunks: {collection.count()}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – SIMILARITY RETRIEVAL
#  Goal: Given a user query, find the Top-K most semantically similar chunks.
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_relevant_chunks(
    query: str,
    embedding_model: HuggingFaceEmbeddingModel,
    collection: Any,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Converts query to vector and retrieves the top-K closest matching document chunks.
    Filters out chunks whose cosine similarity falls below the relevance threshold.
    """
    print(f"\n[STEP 6] Retrieving top-{top_k} relevant chunks for query: '{query}'")

    # 1. Embed query in the same vector space
    query_vector = embedding_model.embed([query])[0]

    # 2. Query vector database for nearest neighbors
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    relevant_docs = []
    print(f"  Retrieved {len(documents)} candidate passages:")
    for rank, (doc, dist) in enumerate(zip(documents, distances), start=1):
        similarity = 1.0 - dist
        snippet = doc[:90].replace("\n", " ")
        is_relevant = similarity >= threshold
        status = "MATCH" if is_relevant else "LOW CONFIDENCE (Ignored)"
        print(f"    [{rank}] Cosine Similarity: {similarity:.3f} [{status}] | \"{snippet}...\"")
        if is_relevant:
            relevant_docs.append(doc)

    if not relevant_docs:
        print("  WARNING: All retrieved passages fell below the similarity threshold.")

    return relevant_docs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – PROMPT AUGMENTATION
#  Goal: Construct an instruction-grounded prompt incorporating retrieved context.
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Synthesizes the prompt enforcing strict document grounding to prevent hallucinations.
    """
    if not context_chunks:
        formatted_context = "No relevant context found in document."
    else:
        formatted_context = "\n\n".join(
            f"[Context Snippet {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )

    prompt = f"""You are an authoritative enterprise AI assistant.
Answer the user's question STRICTLY based on the provided document context snippets.

GROUNDING RULES:
1. Base your answer solely on the context provided below.
2. If the context does not contain the answer, state clearly:
   "I cannot find sufficient information in the provided document to answer this question."
3. Cite the relevant snippet number(s) (e.g. [Context Snippet 1]) to justify your answer.
4. Keep the answer concise, professional, and truthful.

=== DOCUMENT CONTEXT BEGIN ===
{formatted_context}
=== DOCUMENT CONTEXT END ===

USER QUESTION: {query}

FINAL ANSWER:"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – GENERATION (OpenAI API / Local Grounded Answer)
#  Goal: Generate final factual answer from the augmented prompt.
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    prompt: str,
    query: str,
    context_chunks: List[str],
    model: str = OPENAI_MODEL,
    max_tokens: int = MAX_ANSWER_TOKENS
) -> str:
    """
    Generates an answer using OpenAI API if OPENAI_API_KEY is available.
    If no API key is provided, performs clean local grounded extraction.
    If the question is out of scope (no matching context), refuses to hallucinate.
    """
    # Guardrail: If no passages passed the relevance threshold, return standard refusal
    if not context_chunks:
        return "I cannot find sufficient information in the provided document to answer this question."

    api_key = os.environ.get("OPENAI_API_KEY")

    # If OpenAI API Key is provided and SDK is available -> Call OpenAI Chat API
    if api_key and OPENAI_AVAILABLE:
        print(f"\n[STEP 7] Generating answer with OpenAI ({model})...")
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise document QA assistant grounded strictly in context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature ensures deterministic, factual output
                max_tokens=max_tokens
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"  OpenAI API call failed ({e}). Defaulting to local extraction.")

    # Local Grounded Synthesis (Free, instant, deterministic)
    print("\n[STEP 7] Generating grounded answer from retrieved context (Local Engine):")
    
    # Filter common English stopwords so meaningless words don't produce false matches
    STOPWORDS = {
        "what", "is", "the", "a", "an", "and", "or", "in", "of", "to", "how",
        "do", "does", "can", "are", "for", "on", "it", "at", "by", "with", "about"
    }
    query_words = set(w for w in query.lower().replace("?", "").replace(",", "").split() if w not in STOPWORDS)
    best_sentences = []

    for i, chunk in enumerate(context_chunks):
        sentences = [s.strip() for s in chunk.replace("\n", " ").split(".") if len(s.strip()) > 10]
        for s in sentences:
            sentence_words = set(s.lower().split())
            matching_keywords = query_words.intersection(sentence_words)
            if matching_keywords:
                score = len(matching_keywords)
                best_sentences.append((score, f"{s}. (Source: Snippet {i+1})"))

    best_sentences.sort(key=lambda x: x[0], reverse=True)

    if best_sentences:
        top_answers = [item[1] for item in best_sentences[:3]]
        return "\n".join(f"- {ans}" for ans in top_answers)
    else:
        return "I cannot find sufficient information in the provided document to answer this question."


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 – INTERACTIVE Q&A SESSION
#  Goal: Interactive loop enabling continuous document querying.
# ══════════════════════════════════════════════════════════════════════════════

def run_qa_loop(
    embedding_model: HuggingFaceEmbeddingModel,
    collection: Any
) -> None:
    """Runs interactive terminal Q&A loop."""
    print("\n" + "=" * 70)
    print("  RAG Document Q&A Session Active")
    print("  Type your question and press Enter. (Type 'exit' or 'quit' to stop)")
    print("=" * 70)

    while True:
        try:
            user_query = input("\nYour Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break

        if user_query.lower() in {"quit", "exit", "q"}:
            print("Session ended. Goodbye!")
            break

        if not user_query:
            continue

        # 1. Retrieve
        relevant_chunks = retrieve_relevant_chunks(user_query, embedding_model, collection, top_k=TOP_K)

        # 2. Augment
        augmented_prompt = build_prompt(user_query, relevant_chunks)

        # 3. Generate
        answer = generate_answer(augmented_prompt, user_query, relevant_chunks)

        print("\n" + "-" * 70)
        print("ANSWER:")
        print(textwrap.fill(answer, width=70))
        print("-" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 – MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Executes the complete RAG ingestion and retrieval pipeline."""
    print("=" * 70)
    print("  RAG PIPELINE: Retrieval-Augmented Generation Architecture")
    print("=" * 70)

    # Ensure sample PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"'{PDF_PATH}' not found. Generating sample corporate handbook...")
        from create_sample_pdf import generate_handbook
        generate_handbook(PDF_PATH)

    # 1. Text Extraction
    raw_text = extract_text_from_pdf(PDF_PATH)

    # 2. Text Chunking
    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # 3. Embedding Model Setup (Open-source Hugging Face model)
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

    # 4. Vector Database Setup
    _, collection = setup_chromadb(COLLECTION_NAME, VECTOR_DB_DIR)

    # 5. Indexing / Ingestion
    ingest_chunks(chunks, embedding_model, collection)

    # 6. Sample automated query demonstration
    demo_query = "What is the policy for annual leave and sick leave?"
    print("\n" + "#" * 70)
    print(f"  RUNNING DEMO QUERY: '{demo_query}'")
    print("#" * 70)
    top_chunks = retrieve_relevant_chunks(demo_query, embedding_model, collection, top_k=TOP_K)
    prompt = build_prompt(demo_query, top_chunks)
    demo_answer = generate_answer(prompt, demo_query, top_chunks)
    print("\n" + "-" * 70)
    print("DEMO ANSWER:")
    print(demo_answer)
    print("-" * 70)

    # If run in non-interactive terminal (automated test), finish gracefully
    if not sys.stdin.isatty():
        print("\nPipeline executed successfully in automated mode.")
        return

    # 7. Start interactive terminal session
    run_qa_loop(embedding_model, collection)


if __name__ == "__main__":
    main()
