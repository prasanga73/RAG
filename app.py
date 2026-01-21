import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from numpy import dot
from numpy.linalg import norm
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
if "query_encoder" not in st.session_state:
    st.session_state.query_encoder = None
if "doc_encoder" not in st.session_state:
    st.session_state.doc_encoder = None
if "cross_encoder" not in st.session_state:
    st.session_state.cross_encoder = None

# =========================
# 1️⃣ Clean clause text
# =========================
def clean_clause_text(text: str) -> str:
    """Strip whitespace from clause text."""
    return text.strip()


# =========================
# 2️⃣ Keyword extraction
# =========================
LEGAL_STOPWORDS = {
    "the", "of", "and", "for", "in", "on", "to", "with", "under", "by", "a", "an"
}

def extract_keywords(text: str):
    """
    Simple keyword extraction for legal queries:
    - Lowercases text
    - Splits by non-word characters
    - Removes stopwords and very short words
    """
    words = re.findall(r"\b\w+\b", text.lower())
    keywords = [w for w in words if len(w) > 2 and w not in LEGAL_STOPWORDS]
    return set(keywords)


# =========================
# 3️⃣ Hybrid retrieval (vector + keyword)
# =========================
def hybrid_search(query: str, vector_k: int = 10, keyword_k: int = 5):
    """
    Retrieve candidate clauses from vector similarity and keyword matching.
    Returns combined list (vector first, then keyword-only hits).
    """
    vectorstore = st.session_state.vectorstore
    
    # Vector search
    vector_docs = vectorstore.similarity_search(query, k=vector_k)

    # Keyword-only search (for fallback)
    keywords = extract_keywords(query)
    if keywords:
        keyword_docs = vectorstore.similarity_search(" ".join(keywords), k=keyword_k)
    else:
        keyword_docs = []

    # Combine while keeping original order (vector first)
    combined_docs = vector_docs + [d for d in keyword_docs if d not in vector_docs]

    return combined_docs


# =========================
# 4️⃣ Keyword boosting
# =========================
def keyword_boosting(docs, query: str, boost: bool = True):
    """Boost scores for docs containing important keywords."""
    query_keywords = extract_keywords(query)
    scored_docs = []
    for d in docs:
        text = d.page_content.lower()
        keyword_score = sum(1 for w in query_keywords if w in text)
        score = 1.0 + keyword_score if boost else 1.0
        scored_docs.append((score, d))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored_docs]


# =========================
# 5️⃣ Reranking with dual-encoder + cross-encoder
# =========================
def rerank_docs(docs, query: str, alpha: float = 0.5, show_scores: bool = False):
    """
    Rerank candidate clauses using:
    - Bi-encoder similarity (query + doc)
    - Cross-encoder relevance
    Combines scores with weight alpha.
    Filters docs below threshold.
    """
    if not docs:
        return []

    query_encoder = st.session_state.query_encoder
    doc_encoder = st.session_state.doc_encoder
    cross_encoder = st.session_state.cross_encoder

    # Bi-encoder similarity
    query_vec = query_encoder.encode(query)
    doc_vecs = [doc_encoder.encode(d.page_content) for d in docs]
    bi_scores = [dot(query_vec, d_vec)/(norm(query_vec)*norm(d_vec)) for d_vec in doc_vecs]

    # Cross-encoder scores
    pairs = [[query, d.page_content] for d in docs]
    cross_scores = cross_encoder.predict(pairs)

    # Combined score
    combined_scores = [alpha*b + (1-alpha)*c for b, c in zip(bi_scores, cross_scores)]

    if show_scores:
        st.write("### 📊 Reranker Scores")
        for i, (doc, score) in enumerate(zip(docs, combined_scores), 1):
            clause_id = doc.metadata.get("clause_id", "N/A")
            st.caption(f"**{i}. Clause ID:** {clause_id} | **Score:** {score:.4f}")

    # Filter by threshold
    filtered_docs = [d for d, s in zip(docs, combined_scores)]

    # Sort descending
    filtered_docs_sorted = sorted(filtered_docs, key=lambda d: combined_scores[docs.index(d)], reverse=True)

    return filtered_docs_sorted


# =========================
# 6️⃣ Build final context
# =========================
def build_context(docs, top_k: int = 5):
    """Concatenate top-k clauses into clean legal context."""
    docs = docs[:top_k]
    context_blocks = []
    for d in docs:
        text = clean_clause_text(d.page_content)
        legal_source = d.metadata.get("legal_document_source", "")
        meta_block = f"\nLegal Document Source: {legal_source}" if legal_source else ""
        context_blocks.append(f"Clause Text:\n{text}{meta_block}")
    return "\n\n" + ("\n" + "-" * 80 + "\n").join(context_blocks)


# =========================
# 7️⃣ Full improved pipeline
# =========================
def retrieve_context_pipeline(
    query: str,
    top_k: int = 5,
    vector_k: int = 10,
    keyword_k: int = 5,
    keyword_boost: bool = True,
    rerank: bool = True,
    reranker_threshold: float = 0.0,
    alpha: float = 0.5,
    show_scores: bool = False
):
    """
    Full retrieval pipeline combining hybrid search, keyword boosting, and reranking.
    """
    # Step 1: Hybrid retrieval
    docs = hybrid_search(query, vector_k=vector_k, keyword_k=keyword_k)
    if not docs:
        return [], "NO_RELEVANT_CONTEXT"

    # Step 2: Keyword boosting
    docs = keyword_boosting(docs, query, boost=keyword_boost)

    # Step 3: Rerank (optional)
    if rerank:
        docs = rerank_docs(docs, query, alpha=alpha, show_scores=show_scores)
        if not docs:
            return [], "NO_RELEVANT_CONTEXT"

    # Step 4: Build final context
    context = build_context(docs, top_k=top_k)
    return docs, context


# =========================
# 8️⃣ Format response
# =========================
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


# =========================
# Load Models
# =========================
@st.cache_resource
def load_vectorstore():
    """Load the persisted Chroma vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_nepal_law_store",
        embedding_function=embedding_model
    )
    return vectorstore

@st.cache_resource
def load_encoders():
    """Load the dual-encoder and cross-encoder models."""
    query_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return query_encoder, doc_encoder, cross_encoder


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
        
        st.header("⚙️ Retrieval Settings")
        top_k = st.slider("Top-k clauses to return:", 2, 10, 5)
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Load models
    try:
        with st.spinner("Loading legal knowledge base and AI models..."):
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = load_vectorstore()
            
            if st.session_state.query_encoder is None:
                query_enc, doc_enc, cross_enc = load_encoders()
                st.session_state.query_encoder = query_enc
                st.session_state.doc_encoder = doc_enc
                st.session_state.cross_encoder = cross_enc
        
        st.success("✅ Knowledge base and models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
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
                    docs, context = retrieve_context_pipeline(
                        query,
                        top_k=top_k,
                        vector_k=top_k * 2,
                        keyword_k=top_k
                    )
                    
                    response = format_response(query, docs[:top_k])
                    st.markdown(response, unsafe_allow_html=True)
                    
                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    st.write(f"Debug info: {str(e)}")


if __name__ == "__main__":
    main()