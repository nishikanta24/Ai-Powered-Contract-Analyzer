import streamlit as st
import tiktoken  # Import tiktoken
import os
import tempfile  # For temporary file handling

from dotenv import load_dotenv
from chains.rag import ContractRAGChain  # Import the RAG chain class
from chains.analysis import ContractAnalysisChain  # Import the analysis chain class
from utils.file_handler import extract_text  # Text extraction utility
from utils.vector_store import create_vector_store  # Vector store utility
from langchain_core.documents import Document # For creating Document objects

# Load environment variables
load_dotenv()

# Constants for token management
MAX_TOKENS = 8192
WARN_THRESHOLD = 5734  # 70% of max
BUFFER_TOKENS = 500

# Token counting function
encoder = tiktoken.encoding_for_model('gpt-4')  # Adjust model if needed

def count_tokens(messages):
    total_tokens = 0
    for msg in messages:
        total_tokens += len(encoder.encode(msg['content']))
    return total_tokens

# Helper function to get the RAG chain instance
def get_rag_chain(vector_store):
    return ContractRAGChain(vector_store)

# Helper function to get the analysis chain instance
def get_analysis_chain():
    return ContractAnalysisChain()

# Main Streamlit app
st.title("Contract Evaluation Tool MVP")

# Sidebar for chunking configuration
st.sidebar.header("Chunking Configuration")
chunk_size = st.sidebar.slider("Chunk Size (characters)", min_value=500, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", min_value=50, max_value=500, value=200, step=50)

# File upload section
uploaded_files = st.file_uploader("Upload PDF or DOCX contracts", accept_multiple_files=True, type=["pdf", "docx"])

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        # Get the file extension from the uploaded file name
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save uploaded file to a temporary location with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        extracted_text = extract_text(tmp_path)
        if extracted_text:
            doc = Document(page_content=extracted_text, metadata={"source": uploaded_file.name})
            all_docs.append(doc)
        else:
            st.warning(f"Failed to extract text from {uploaded_file.name}. Skipping.")
        
        # Clean up temporary file
        os.unlink(tmp_path)
    
    if all_docs:
        with st.spinner("Processing documents and creating embeddings..."):
            # Create vector store with chunking parameters
            vector_store = create_vector_store(
                all_docs, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            st.session_state['vector_store'] = vector_store
        
        # Display chunking information
        total_chunks = vector_store.index.ntotal
        st.info(f"📄 Processed {len(all_docs)} documents into {total_chunks} chunks (Size: {chunk_size}, Overlap: {chunk_overlap})")
        
        with st.spinner("Analyzing contracts..."):
            # Analyze contracts (still use original documents for full analysis)
            analysis_chain = get_analysis_chain()
            analysis_results = []
            for doc in all_docs:
                result = analysis_chain.analyze_contract(doc.page_content)
                analysis_results.append({"source": doc.metadata["source"], "result": result})
            st.session_state['analysis_results'] = analysis_results
        
        st.success("Contracts processed and analyzed successfully!")
    else:
        st.error("No valid text extracted from uploaded files.")

# Display analysis results if available
if 'analysis_results' in st.session_state:
    st.header("Analysis Results")
    for item in st.session_state['analysis_results']:
        st.subheader(f"Results for {item['source']}")
        
        # Display summary and overall risk score prominently
        result = item['result']
        if 'overall_risk_score' in result and result['overall_risk_score']:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Summary:**", result.get('summary', 'N/A'))
            with col2:
                risk_score = result['overall_risk_score']
                risk_color = "🔴" if risk_score >= 7 else "🟡" if risk_score >= 4 else "🟢"
                st.metric("Risk Score", f"{risk_score}/10 {risk_color}")
        
        # Show detailed results in expandable section
        with st.expander("View Detailed Analysis"):
            st.json(result)

# Chatbot section
st.subheader("Contract Q&A Chatbot")

if 'vector_store' in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the contracts:"):
        # Calculate tokens
        new_messages = st.session_state.messages + [{"role": "user", "content": prompt}]
        total_used_tokens = count_tokens(new_messages) + BUFFER_TOKENS

        if total_used_tokens > MAX_TOKENS:
            st.error(f"Token limit exceeded ({total_used_tokens}/{MAX_TOKENS}). Please start a new chat.")
        else:
            if total_used_tokens > WARN_THRESHOLD:
                st.warning(f"High token usage ({total_used_tokens}/{MAX_TOKENS}). Consider summarizing or starting a new chat.")

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.spinner("Searching through contract chunks..."):
                rag_chain = get_rag_chain(st.session_state['vector_store'])
                response = rag_chain.invoke(prompt)

            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    # Context management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Summarize Chat & Continue"):
            st.info("This feature is to be implemented.")
    with col2:
        if st.button("Start New Chat"):
            st.session_state.messages = []
            st.rerun()  # Updated from experimental_rerun
else:
    st.info("Upload and analyze contracts first.")

# Add help section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("💡 Tips")
st.sidebar.markdown("""
**Chunking Settings:**
- **Smaller chunks** (500-800): Better for specific details
- **Larger chunks** (1200-2000): Better for context
- **More overlap** (300-400): Better continuity between chunks

**Best Questions:**
- "What are the payment terms?"
- "Are there any liability clauses?"
- "What happens if the contract is terminated?"
- "What are the key risks?"
""")