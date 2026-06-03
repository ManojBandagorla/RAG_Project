
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

upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    # Extract PDF text
    raw_text = extract_text(temp_file_path)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(raw_text)

    # Create Vector Store
    with st.spinner("Indedocument..."):

        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_index"
        )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a document summarization assistant.

        Context:
        {context}

        Question:
        {input}

        Provide a concise summary covering key topics.
        """
    )

    # Document Chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Retrieval Chain
    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    summary_prompt = "Please summarize this document based on the key topics."

    with st.spinner("Running RAG summarization..."):

        result = rag_chain.invoke({
            "input": summary_prompt
        })

    st.subheader("Summary")
    st.write(result["answer"])