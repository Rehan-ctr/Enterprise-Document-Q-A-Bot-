"""
smoke_test.py
-------------
Validates every component of the RAG pipeline step-by-step:
  [1] PDF text extraction      (pypdf / PyMuPDF)
  [2] Text chunking            (sliding window with overlap)
  [3] Embedding model load     (Hugging Face Transformers / torch)
  [4] Vector DB setup          (persistent collection creation)
  [5] Chunk ingestion          (embed + upsert)
  [6] Similarity retrieval     (cosine nearest-neighbor search)
  [7] Prompt construction      (grounding template assembly)
  [8] Grounded answer test     (factual retrieval answer generation)

Run: py -3.11 smoke_test.py
"""

import sys
import os
import traceback

from rag_pipeline import (
    extract_text_from_pdf,
    chunk_text,
    load_embedding_model,
    setup_chromadb,
    ingest_chunks,
    retrieve_relevant_chunks,
    build_prompt,
    generate_answer,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SEP  = "=" * 65

def run_test(name: str, fn):
    """Runs a single test function and reports pass/fail."""
    print(f"\n{SEP}")
    print(f"  RUNNING TEST: {name}")
    print(SEP)
    try:
        result = fn()
        print(f"{PASS} {name}")
        return result, True
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"  Exception: {e}")
        traceback.print_exc()
        return None, False


# ── TEST 1: PDF Extraction ────────────────────────────────────────────────────
def test_pdf_extraction():
    if not os.path.exists("handbook.pdf"):
        from create_sample_pdf import generate_handbook
        generate_handbook("handbook.pdf")

    text = extract_text_from_pdf("handbook.pdf")
    assert isinstance(text, str), "Expected a string"
    assert len(text) > 100, "Text too short — PDF may be empty"
    assert "leave" in text.lower(), "Expected corporate handbook content"
    print(f"  Extracted {len(text):,} characters from PDF.")
    return text


# ── TEST 2: Text Chunking ─────────────────────────────────────────────────────
def test_chunking(raw_text):
    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    assert isinstance(chunks, list), "Expected a list of chunks"
    assert len(chunks) > 0, "No chunks produced"
    assert all(isinstance(c, str) for c in chunks), "All chunks must be strings"
    print(f"  {len(chunks)} chunks produced. First chunk preview:")
    print(f"  \"{chunks[0][:100]}...\"")
    return chunks


# ── TEST 3: Embedding Model ───────────────────────────────────────────────────
def test_embedding_model():
    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    vectors = model.embed(["This is a test document sentence.", "What is annual leave?"])
    assert isinstance(vectors, list), "Expected list of vector embeddings"
    assert len(vectors) == 2, "Expected 2 embeddings"
    dim = len(vectors[0])
    assert dim == 384, f"Expected 384-dimensional vector, got {dim}"
    
    # Verify unit normalization: ||v|| ~ 1.0
    norm = sum(x**2 for x in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3, f"Vector not unit-normalized: norm={norm}"
    print(f"  Embedding dimensions: {dim} | L2 norm: {norm:.4f} (unit sphere verified)")
    return model


# ── TEST 4: Vector Database Setup ─────────────────────────────────────────────
def test_vector_db():
    client, collection = setup_chromadb("smoke_test_collection", "./smoke_test_vector_db")
    assert collection is not None, "Collection object is None"
    print(f"  Vector database initialized. Item count: {collection.count()}")
    return client, collection


# ── TEST 5: Ingestion ─────────────────────────────────────────────────────────
def test_ingestion(chunks, model, collection):
    test_chunks = chunks[:5]
    ingest_chunks(test_chunks, model, collection, batch_size=5)
    stored = collection.count()
    assert stored >= len(test_chunks), f"Expected at least {len(test_chunks)} stored vectors, got {stored}"
    print(f"  {stored} vectors verified in Vector Database.")


# ── TEST 6: Similarity Retrieval ─────────────────────────────────────────────
def test_retrieval(model, collection):
    query = "How many days of annual leave do employees get?"
    results = retrieve_relevant_chunks(
        query=query,
        embedding_model=model,
        collection=collection,
        top_k=min(TOP_K, collection.count())
    )
    assert isinstance(results, list), "Expected a list of retrieved chunks"
    assert len(results) > 0, "No results returned from query"
    assert any("leave" in doc.lower() for doc in results), "Failed to retrieve leave policy passage"
    print(f"  Successfully retrieved top-{len(results)} relevant passages matching query.")
    return results


# ── TEST 7: Prompt Construction ───────────────────────────────────────────────
def test_prompt(retrieved_chunks):
    query = "What is the annual leave entitlement?"
    prompt = build_prompt(query, retrieved_chunks)
    assert "DOCUMENT CONTEXT" in prompt, "Missing context demarcation"
    assert query in prompt, "Query missing from prompt"
    assert "Context Snippet 1" in prompt, "Snippet citation numbering missing"
    print(f"  Prompt constructed successfully ({len(prompt)} chars).")


# ── TEST 8: Grounded Answer Generation ────────────────────────────────────────
def test_generation(retrieved_chunks):
    query = "What is the annual leave entitlement?"
    prompt = build_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt, query, retrieved_chunks)
    assert isinstance(answer, str) and len(answer) > 10, "Empty or invalid answer"
    print(f"  Generated grounded answer:\n  {answer}")


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────────────
def main():
    print("\n" + SEP)
    print("  RAG PIPELINE END-TO-END SMOKE TEST")
    print("  Validates: Extraction -> Chunking -> Embeddings -> VectorDB -> Search -> QA")
    print(SEP)

    results = {}

    # Test 1
    text, ok = run_test("PDF Extraction", test_pdf_extraction)
    results["PDF Extraction"] = ok
    if not ok:
        sys.exit(1)

    # Test 2
    chunks, ok = run_test("Text Chunking", lambda: test_chunking(text))
    results["Text Chunking"] = ok

    # Test 3
    model, ok = run_test("Hugging Face Embedding Model Load", test_embedding_model)
    results["Embedding Model"] = ok
    if not ok:
        sys.exit(1)

    # Test 4
    db_result, ok = run_test("Vector Database Setup", test_vector_db)
    results["Vector DB Setup"] = ok
    if not ok:
        sys.exit(1)
    client, collection = db_result

    # Test 5
    _, ok = run_test("Passage Ingestion & Upsert", lambda: test_ingestion(chunks, model, collection))
    results["Chunk Ingestion"] = ok

    # Test 6
    retrieved, ok = run_test("Semantic Similarity Retrieval", lambda: test_retrieval(model, collection))
    results["Similarity Retrieval"] = ok

    # Test 7
    _, ok = run_test("Prompt Construction", lambda: test_prompt(retrieved))
    results["Prompt Construction"] = ok

    # Test 8
    _, ok = run_test("Grounded Answer Generation", lambda: test_generation(retrieved))
    results["Answer Generation"] = ok

    # Clean up smoke test directory
    try:
        import shutil
        if os.path.exists("./smoke_test_vector_db"):
            shutil.rmtree("./smoke_test_vector_db", ignore_errors=True)
    except Exception:
        pass

    # ── SUMMARY REPORT ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FINAL SMOKE TEST SUMMARY")
    print(SEP)
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {test_name}")
        if not passed:
            all_passed = False

    print(SEP)
    if all_passed:
        print("\n  ALL 8 TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
        print("  The entire RAG architecture is verified and ready for production/interview.")
    else:
        print("\n  SOME TESTS ENCOUNTERED ERRORS. Check traceback above.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
