from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

def build_rag_chain(llm, retriever):

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions using the provided documents.

Use ONLY the context below.
If the answer is not available, say "I don't know".

Context:
{context}

Question:
{input}

Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, document_chain)