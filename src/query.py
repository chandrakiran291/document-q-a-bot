"""
query.py

Query pipeline for the RAG Q&A bot.

Responsibilities:
1. Embed the user's natural-language question using the same embedding model
   used during ingestion (text-embedding-004) so vectors are comparable.
2. Retrieve the top-k most similar chunks from the persisted ChromaDB collection.
3. Filter out chunks that are too dissimilar (distance above threshold) to be
   genuinely relevant.
4. Build a strictly-grounded prompt (with inline source citations) and send it
   to the Gemini generation model.
5. Return the answer text alongside the citation list and raw retrieved chunks.
"""

import os
import sys

import google.generativeai as genai
import chromadb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from embeddings import GeminiEmbeddingFunction

genai.configure(api_key=config.GEMINI_API_KEY)

# Module-level cache so Streamlit re-runs don't reconnect to the DB on every keystroke
_collection = None


def get_collection():
    """
    Loads (and caches) the persisted ChromaDB collection. Raises a clear error
    if ingestion hasn't been run yet, since query.py never creates the collection.
    """
    global _collection
    if _collection is not None:
        return _collection

    if not os.path.isdir(config.DB_DIR) or not os.listdir(config.DB_DIR):
        raise RuntimeError(
            f"No vector database found at {config.DB_DIR}. "
            "Run `python src/ingest.py` first to index your documents."
        )

    client = chromadb.PersistentClient(path=config.DB_DIR)
    embedding_fn = GeminiEmbeddingFunction(
        api_key=config.GEMINI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
        task_type="retrieval_query",
    )

    try:
        _collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load collection '{config.COLLECTION_NAME}' from {config.DB_DIR}. "
            f"Did ingestion complete successfully? Original error: {e}"
        )

    return _collection


# ---------------------------------------------------------------------------
# Step 5: Similarity Search & Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(user_query: str, k: int = config.TOP_K) -> list[dict]:
    """
    Embeds the query and retrieves the top-k closest chunks from ChromaDB.
    Filters out chunks whose distance exceeds config.DISTANCE_THRESHOLD,
    since those are not meaningfully related to the question.

    Returns a list of dicts: {"text": ..., "metadata": ..., "distance": ...}
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[user_query],
        n_results=k,
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist <= config.DISTANCE_THRESHOLD:
            retrieved.append({"text": doc, "metadata": meta, "distance": dist})

    return retrieved


# ---------------------------------------------------------------------------
# Step 6: Prompt Engineering & Answer Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a professional, accurate document Q&A assistant. "
    "Answer the user's question using ONLY the provided document context below. "
    "Cite the sources (filenames and pages) inline next to facts you mention, "
    "in the form (filename, Page X). "
    "If the answer cannot be found in the context, clearly state: "
    "'I am sorry, but the provided documents do not contain the answer to your question.' "
    "Do not make up facts or use external knowledge sources."
)


def build_prompt(user_query: str, retrieved_chunks: list[dict]) -> str:
    """Assembles the grounded prompt from retrieved context blocks."""
    context_blocks = []
    for chunk in retrieved_chunks:
        source_name = chunk["metadata"]["source"]
        page_num = chunk["metadata"]["page"]
        citation_str = f"Source: {source_name}, Page: {page_num}"
        context_blocks.append(f"[{citation_str}]\nContext: {chunk['text']}")

    context_payload = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )
    return prompt


def query_rag_pipeline(user_query: str, k: int = config.TOP_K) -> dict:
    """
    Full RAG query: retrieve -> build grounded prompt -> generate answer.

    Returns:
        {
            "answer": str,
            "citations": list[str],
            "raw_context": list[dict]   # full chunk dicts with text/metadata/distance
        }
    """
    retrieved_chunks = retrieve_relevant_chunks(user_query, k=k)

    if not retrieved_chunks:
        return {
            "answer": "I am sorry, but the provided documents do not contain the answer to your question.",
            "citations": [],
            "raw_context": [],
        }

    prompt = build_prompt(user_query, retrieved_chunks)

    model = genai.GenerativeModel(config.GENERATION_MODEL)
    response = model.generate_content(prompt)

    citations = [
        f"{c['metadata']['source']}, Page {c['metadata']['page']}"
        for c in retrieved_chunks
    ]
    # De-duplicate citations while preserving order (same doc/page can appear via multiple chunks)
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    return {
        "answer": response.text,
        "citations": unique_citations,
        "raw_context": retrieved_chunks,
    }


# ---------------------------------------------------------------------------
# CLI smoke test: python src/query.py "your question"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.validate_config()
    if len(sys.argv) < 2:
        print('Usage: python src/query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = query_rag_pipeline(question)

    print("\nANSWER:\n" + result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c}")
