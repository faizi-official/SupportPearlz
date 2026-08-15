import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List
from src.ingestion.loaders import DocumentIngestionPipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SupportPearlz AI | Enterprise Support Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED MODERN UI STYLING & TYPOGRAPHY ---
st.markdown("""
    <style>
    /* Main Background & Global Font Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Optimized Header Styles with Clear Hierarchy */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0.75rem;
        letter-spacing: -0.025em;
    }
    
    .main-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Typography Overrides for Cleaner Headings */
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 10px;
        color: #38bdf8 !important;
        font-weight: 600;
        border: 1px solid #334155;
    }

    /* Metric Cards Custom Design */
    .stMetric {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
    <div class="main-header">
        <div class="main-title">🛡️ SupportPearlz Enterprise AI</div>
        <div class="main-subtitle">Your intelligent customer support agent built on custom knowledge bases. Powered by Advanced RAG, LangChain, Pydantic & OpenAI.</div>
    </div>
""", unsafe_allow_html=True)

# --- PATH CONFIGURATIONS ---
VECTOR_STORE_PATH = "data/vector_store"
KB_PATH = "data/knowledge_base"
os.makedirs(KB_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/customer-support.png", width=90)
    st.markdown("### ⚙️ Control Panel")
    
    # API Key Input
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    api_key = user_api_key or os.getenv("OPENAI_API_KEY")
    
    st.markdown("---")
    
    # Model Configuration
    st.markdown("### 🧠 Model Settings")
    selected_model = st.selectbox("Select LLM Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    st.markdown("---")
    
    # Live Document Uploader & Physical Saving Logic
    st.markdown("### 📁 Knowledge Base")
    uploaded_files = st.file_uploader("Upload reference documents", accept_multiple_files=True, type=["txt", "pdf", "md"])
    
    if uploaded_files:
        saved_count = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(KB_PATH, uploaded_file.name)
            # Physical disk writing check
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
        
        if saved_count > 0:
            st.success(f"Successfully saved {saved_count} file(s) to knowledge base!")
            
if st.button("🔄 Rebuild Vector Store", use_container_width=True):
        st.cache_resource.clear()
        import shutil
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        st.success("Vector store cache cleared! Rebuilding...")
        st.rerun()

    st.markdown("---")
    
    # Session Management
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()            
    
    st.markdown("---")
    
    # Session Management
    if st.button("🗑️ Clear Chat History", use_content_width=True if hasattr(st, "use_content_width") else True):
        st.session_state.messages = []
        st.rerun()

# --- PYDANTIC SCHEMA ---
class SupportResponse(BaseModel):
    answer: str = Field(description="The detailed, professional answer to the user's question based strictly on the context.")
    sources: List[str] = Field(description="List of source document names used to formulate the answer.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on context relevance.")
    answered: bool = Field(description="True if the question could be answered from the context, false otherwise.")

# --- VECTOR STORE CACHING & INITIALIZATION ---
@st.cache_resource
def get_or_create_vector_store(api_key_str: str):
    if not api_key_str:
        return None
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key_str
    )
    
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        if not os.path.exists(KB_PATH) or not os.listdir(KB_PATH):
            return None
        
        pipeline = DocumentIngestionPipeline(kb_path=KB_PATH)
        docs = pipeline.load_documents()
        if not docs:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = text_splitter.split_documents(docs)
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        return vector_store
    else:
        return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

# --- MAIN APPLICATION LOGIC ---
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to initialize the support assistant.")
else:
    try:
        with st.spinner("🔄 Initializing Knowledge Base & Vector Store..."):
            vector_store = get_or_create_vector_store(api_key)
            
        if vector_store is None:
            st.error(f"❌ No documents found in the '{KB_PATH}' directory. Please upload files via the sidebar or add them manually.")
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            llm = ChatOpenAI(model=selected_model, temperature=temperature, openai_api_key=api_key)
            structured_llm = llm.with_structured_output(SupportResponse)
            
            # Query Condensation Prompt
            condense_prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                ("user", "Given the above conversation, generate a standalone search query that can be used to retrieve relevant information from the knowledge base. Only output the query string.")
            ])
            condense_chain = condense_prompt | llm | StrOutputParser()

            # Main RAG Answering Prompt
            template = """You are the official enterprise customer support AI assistant for SupportPearlz. Provide a professional, highly accurate answer to the user's question based strictly on the provided context. If the answer cannot be found within the context, set 'answered' to false and give a polite response explaining that information is missing.

Context:
{context}

Question: {question}"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in docs])

            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display Chat History with Avatars
            for message in st.session_state.messages:
                avatar = "🛡️" if message["role"] == "assistant" else "👤"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            # Chat Input
            if query := st.chat_input("Type your support question here... (e.g., How do I reset my password?)"):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(query)

                with st.chat_message("assistant", avatar="🛡️"):
                    with st.spinner("Analyzing context & formulating response..."):
                        # Prepare chat history for LangChain
                        chat_history = []
                        for msg in st.session_state.messages[:-1]:
                            if msg["role"] == "user":
                                chat_history.append(HumanMessage(content=msg["content"]))
                            else:
                                chat_history.append(AIMessage(content=msg["content"]))
                        
                        # Condense query if chat history exists
                        if len(chat_history) > 0:
                            search_query = condense_chain.invoke({"chat_history": chat_history, "input": query})
                        else:
                            search_query = query

                        # Retrieve and execute chain
                        retrieved_docs = retriever.invoke(search_query)
                        context_text = format_docs(retrieved_docs)
                        
                        chain = prompt | structured_llm
                        structured_response: SupportResponse = chain.invoke({"context": context_text, "question": search_query})
                        
                        # Display Main Answer
                        st.markdown(structured_response.answer)
                        
                        # Advanced Interactive Metadata Panel
                        with st.expander("📊 Advanced Telemetry & Sources Metadata"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(label="Answered from KB", value="Yes" if structured_response.answered else "No")
                            with col2:
                                st.metric(label="Confidence Score", value=f"{structured_response.confidence * 100:.1f}%")
                            with col3:
                                st.metric(label="Sources Retrieved", value=len(structured_response.sources))
                            
                            st.markdown(f"**Standalone Search Query:** `{search_query}`")
                            st.markdown("**Referenced Source Files:**")
                            for src in structured_response.sources:
                                st.code(src, language="text")
                            
                st.session_state.messages.append({"role": "assistant", "content": structured_response.answer})

    except Exception as e:
        st.error(f"⚠️ A technical error occurred: {str(e)}")
        
