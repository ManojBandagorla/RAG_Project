import streamlit as st
import tempfile

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.pdf_loader import load_pdf_text
from ingestion.chunking import get_chunker, chunk_text
from llm.llm_connect import get_llm, get_embeddings
from retrieval.vector_store import create_vectorstore
from retrieval.retriever import get_retriever
from chains.rag_chain import build_rag_chain


st.set_page_config(page_title="RAG Semantic Chatbot")
st.title("RAG-powered PDF Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# -------------------------
# PROCESS FILES
# -------------------------
if uploaded_files :

    embeddings = get_embeddings()
    chunker = get_chunker(embeddings)

    all_docs = []

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        text = load_pdf_text(path)

        if not text or len(text.strip()) < 50:
            continue

        docs = chunk_text(chunker, text)

        if not docs:
            continue

        for doc in docs:
            doc.metadata = {"source": file.name}

        all_docs.extend(docs)

    if not all_docs:
        st.error("No valid documents found")
        st.stop()

    with st.spinner("Indexing..."):
        vectordb = create_vectorstore(all_docs, embeddings)

    llm = get_llm()
    retriever = get_retriever(vectordb)

    st.session_state.rag_chain = build_rag_chain(llm, retriever)

    st.success("Indexing complete!")


# -------------------------
# CHAT UI
# -------------------------
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.rag_chain is None:
        st.error("Please index documents first")

    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke({"input": query})
                answer = result["answer"]
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })