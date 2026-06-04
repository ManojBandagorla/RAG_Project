from langchain_experimental.text_splitter import SemanticChunker

def get_chunker(embeddings):
    return SemanticChunker(embeddings=embeddings)

def chunk_text(chunker, text):
    return chunker.create_documents([text])