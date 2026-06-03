
import requests

original_get = requests.get

def patched_get(*args, **kwargs):
    kwargs["verify"] = False
    return original_get(*args, **kwargs)

requests.get = patched_get

import streamlit as st
from pdfminer.high_level import extract_text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import tempfile 
import os 
import httpx 
client = httpx.Client(verify=False) 

 
# LLM and Embedding setup 
llm = ChatOpenAI( 
   base_url="https://genailab.tcs.in", 
   model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
   api_key="sk-OHahpUWaFfOIcJZflRMF6w", 
   http_client=client 
) 
embedding_model = OpenAIEmbeddings( 
   base_url="https://genailab.tcs.in", 
   model="azure/genailab-maas-text-embedding-3-large", 
   api_key="sk-OHahpUWaFfOIcJZflRMF6w", 
   http_client=client) 
st.set_page_config(page_title="RAG PDF Summarizer")
st.title("RAG-powered PDF Summarizer")
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_text = []

    # Extract text from all PDFs
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        text = extract_text(path)
        all_text.append(text)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []

    for doc_text in all_text:
        chunks = text_splitter.split_text(doc_text)
        all_chunks.extend(chunks)

    # Create Vector Store
    with st.spinner("Indexing documents..."):

        vectordb = Chroma.from_texts(
            texts=all_chunks,
            embedding=embedding_model,
            persist_directory="./multi_doc_index"
        )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that answers questions using the provided documents.

Use ONLY the context below.
If the answer is not available, say "I don't know".

Context:
{context}

Question:
{input}

Answer:
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    user_query = st.chat_input(
        "Ask something from your documents"
    )

    if user_query:

        # User message
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Running RAG..."):

                result = rag_chain.invoke({
                    "input": user_query
                })

                answer = result["answer"]

            st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )