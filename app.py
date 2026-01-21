
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
import os

# Page configuration
st.set_page_config(
    page_title="Nepal Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stChatMessage { border-radius: 10px; }
    .header-text { color: #1f4788; font-size: 2.5rem; font-weight: bold; }
    .subheader-text { color: #555; font-size: 1.1rem; }
    .clause-box { background-color: #f0f2f6; padding: 1.5rem; border-left: 4px solid #1f4788; border-radius: 5px; margin: 1.5rem 0; }
    .info-box { background-color: #e3f2fd; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .clause-item { 
        background-color: #f9fafb; 
        padding: 1.2rem; 
        border-left: 4px solid #1f4788; 
        border-radius: 5px; 
        margin: 1.5rem 0;
        font-size: 0.95rem;
    }
    .clause-source {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .response-text {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "reranker" not in st.session_state:
    st.session_state.reranker = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

@st.cache_resource
def load_vectorstore():
    """Load the persisted Chroma vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_nepal_law_store",
        embedding_function=embedding_model
    )
    return vectorstore, embedding_model

@st.cache_resource
def load_reranker():
    """Load the CrossEncoder reranker."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def clean_clause_text(text: str) -> str:
    """Strip whitespace from clause text."""
    return text.strip()

def retrieve_context(query: str, k: int = 5, keyword_boost: bool = True, rerank: bool = True, relevance_threshold: float = 0.5):
    """
    Retrieve top-k relevant clauses for a query from the vectorstore, with optional
    keyword boosting and reranking. Returns fallback message if no clauses meet
    the relevance threshold.
    
    Parameters:
        query (str): The user query.
        k (int): Number of clauses to return.
        keyword_boost (bool): Whether to boost docs containing query keywords.
        rerank (bool): Whether to rerank results using the reranker.
        relevance_threshold (float): Minimum relevance score (0-1) to include results.
                                     Lower = more lenient, higher = stricter.
        
    Returns:
        list: Top-k relevant clauses, or empty list if no relevant clauses found.
    """
    vectorstore = st.session_state.vectorstore
    reranker = st.session_state.reranker
    
    # Step 1: Initial vector search
    docs = vectorstore.similarity_search(query, k=10)
    if not docs:
        return []

    # Step 2: Optional keyword boosting
    query_words = set(query.lower().split())
    scored_docs = []
    for d in docs:
        text = d.page_content.lower()
        keyword_score = sum(1 for w in query_words if w in text)
        score = 1.0 + keyword_score if keyword_boost else 1.0
        scored_docs.append((score, d))

    # Sort descending by score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    docs_sorted = [d for _, d in scored_docs]

    # Step 3: Optional reranking with relevance threshold
    if rerank:
        # Prepare query-doc pairs for reranker
        pairs = [[query, d.page_content] for d in docs_sorted]
        scores = reranker.predict(pairs)  # Returns scores typically 0-1

        # Filter by relevance threshold BEFORE sorting
        filtered_pairs = [(d, score) for d, score in zip(docs_sorted, scores) if score >= relevance_threshold]
        
        if not filtered_pairs:
            return []
        
        # Sort by reranker scores
        reranked = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)
        docs_sorted = [d for d, _ in reranked]

    # Step 4: Keep only top-k clauses
    docs_sorted = docs_sorted[:k]

    return docs_sorted

def format_response(query: str, docs) -> str:
    """Format the response with context and sources."""
    if not docs:
        return "I do not find a relevant clause in the provided legal texts."
    
    response = f"Based on Nepal law, here is the relevant information for your query:\n\n"
    
    for i, doc in enumerate(docs, 1):
        text = clean_clause_text(doc.page_content)
        legal_source = doc.metadata.get("legal_document_source", "Unknown")
        response += f"**Clause {i}:**\n{text}\n\n**Source:** _{legal_source}_\n\n---\n\n"
    
    response += "**Disclaimer:** This information is based on the legal documents in our knowledge base. For legal advice, please consult a qualified lawyer."
    
    return response

def main():
    # Header
    st.markdown('<p class="header-text">⚖️ Nepal Law Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">AI-powered legal information based on Nepal law</p>', unsafe_allow_html=True)
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This assistant provides information about Nepal law based on:
            - National Criminal Code, 2017 AD
            - Constitution of Nepal
            - Electronic Transactions Act, 2008
            - Other legal documents
            
            **Note:** This is for informational purposes only and not a substitute for legal advice.
            """
        )
        
        st.header("⚙️ Settings")
        num_results = st.slider("Number of clauses to retrieve:", 2, 8, 5)
        keyword_boost = st.checkbox("Enable keyword boosting", value=True)
        rerank = st.checkbox("Enable reranking", value=True)
        relevance_threshold = st.slider("Relevance threshold (0.0 - 1.0):", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        st.caption("⚠️ Higher threshold = stricter filtering, fewer results")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Load models
    try:
        with st.spinner("Loading legal knowledge base..."):
            if st.session_state.vectorstore is None:
                vectorstore, embedding_model = load_vectorstore()
                st.session_state.vectorstore = vectorstore
                st.session_state.embedding_model = embedding_model
            
            if st.session_state.reranker is None:
                st.session_state.reranker = load_reranker()
        
        st.success("✅ Knowledge base loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {e}")
        st.info("Please ensure the Chroma vector store exists at `./chroma_nepal_law_store/`")
        return

    st.divider()

    # Chat interface
    st.subheader("💬 Ask About Nepal Law")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field
    query = st.chat_input("Ask a question about Nepal law...")

    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching legal documents..."):
                try:
                    docs = retrieve_context(
                        query,
                        k=num_results,
                        keyword_boost=keyword_boost,
                        rerank=rerank,
                        relevance_threshold=relevance_threshold
                    )
                    
                    response = format_response(query, docs)
                    st.markdown(response, unsafe_allow_html=True)
                    
                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()