from langchain_chroma import Chroma

def create_vectorstore(docs, embeddings):
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./multi_doc_index"
    )