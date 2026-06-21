"""
main.py

Streamlit web UI for the Document Q&A RAG bot.

"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from query import query_rag_pipeline, get_collection


# Page setup


st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    .citation-pill {
        display: inline-block;
        background-color: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 999px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        font-family: monospace;
    }
    .source-box {
        background-color: #f8fafc;
        border-left: 3px solid #94a3b8;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Document Q&A Bot")
st.caption("Ask questions about your documents. Answers are grounded strictly in retrieved context — no hallucinated facts.")


# Sidebar: status + document list


with st.sidebar:
    st.header("Knowledge Base")

    try:
        config.validate_config()
        api_key_ok = True
    except EnvironmentError as e:
        api_key_ok = False
        st.error(str(e))

    if os.path.isdir(config.DATA_DIR):
        docs = sorted(f for f in os.listdir(config.DATA_DIR) if f.lower().endswith((".pdf", ".docx")))
        if docs:
            st.write(f"**{len(docs)} document(s) in `/data`:**")
            for d in docs:
                st.write(f"- {d}")
        else:
            st.warning("No documents found in `/data`. Add .pdf or .docx files and run ingest.py.")
    else:
        st.warning("`/data` directory not found.")

    st.divider()

    db_ready = os.path.isdir(config.DB_DIR) and len(os.listdir(config.DB_DIR)) > 0
    if db_ready:
        st.success("Vector database loaded ✅")
    else:
        st.error("Vector database not found. Run:\n\n`python src/ingest.py`")

    st.divider()
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=config.TOP_K)

    if st.button("🔄 Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# Chat state


if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            pills = "".join(f'<span class="citation-pill">{c}</span>' for c in msg["citations"])
            st.markdown(pills, unsafe_allow_html=True)
            with st.expander("View retrieved source chunks"):
                for chunk in msg.get("raw_context", []):
                    meta = chunk["metadata"]
                    st.markdown(
                        f'<div class="source-box"><b>{meta["source"]}</b> '
                        f'(Page {meta["page"]}, distance={chunk["distance"]:.3f})<br>{chunk["text"][:400]}...</div>',
                        unsafe_allow_html=True,
                    )

# Chat input


user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    if not api_key_ok:
        st.error("Cannot query: GEMINI_API_KEY is missing. Add it to your .env file.")
    elif not db_ready:
        st.error("Cannot query: vector database not found. Run `python src/ingest.py` first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = query_rag_pipeline(user_question, k=top_k)
                    answer = result["answer"]
                    citations = result["citations"]
                    raw_context = result["raw_context"]
                except Exception as e:
                    answer = f"An error occurred while processing your question: {e}"
                    citations = []
                    raw_context = []

            st.markdown(answer)
            if citations:
                pills = "".join(f'<span class="citation-pill">{c}</span>' for c in citations)
                st.markdown(pills, unsafe_allow_html=True)
                with st.expander("View retrieved source chunks"):
                    for chunk in raw_context:
                        meta = chunk["metadata"]
                        st.markdown(
                            f'<div class="source-box"><b>{meta["source"]}</b> '
                            f'(Page {meta["page"]}, distance={chunk["distance"]:.3f})<br>{chunk["text"][:400]}...</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "raw_context": raw_context,
        })
