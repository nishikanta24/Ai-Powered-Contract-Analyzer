import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Helper method to return Hugging Face configured embeddings instance
# Requires this env var to be loaded globally:
# EMBEDDING_MODEL_NAME

def _get_huggingface_embeddings_model():
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

# Helper method to chunk documents into smaller pieces
def _chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        docs: List of Document objects
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    for doc in docs:
        # Split the document content
        chunks = text_splitter.split_text(doc.page_content)
        
        # Create new Document objects for each chunk
        for i, chunk in enumerate(chunks):
            # Preserve original metadata and add chunk info
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            chunked_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            chunked_docs.append(chunked_doc)
    
    return chunked_docs

# Create a vector store using Hugging Face embeddings by default with chunking
# docs: List of Documents
# persist_path: Directory to save the vector DB locally (optional)
# embedding_model: Optional override for embedding model
# chunk_size: Size of text chunks (default: 1000 characters)
# chunk_overlap: Overlap between chunks (default: 200 characters)

def create_vector_store(docs, persist_path=None, embedding_model=None, chunk_size=1000, chunk_overlap=200):
    if embedding_model is None:
        embedding_model = _get_huggingface_embeddings_model()
    
    # Chunk the documents for better retrieval
    chunked_docs = _chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Create vector store from chunked documents
    vector_store = FAISS.from_documents(chunked_docs, embedding_model)
    
    if persist_path is not None:
        vector_store.save_local(persist_path)
    
    return vector_store

# Load saved vector store from disk with same Hugging Face config

def load_vector_store(persist_path):
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"Persist path does not exist: {persist_path}")
    embedding_model = _get_huggingface_embeddings_model()
    vector_store = FAISS.load_local(persist_path, embedding_model)
    return vector_store

# Return retriever interface for RAG given vector store

def get_retriever(vector_store, search_kwargs=None):
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    return vector_store.as_retriever(search_kwargs=search_kwargs)