import os
import sys
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings
from src.ingestion.loaders import DocumentIngestionPipeline
from src.utils.logging_setup import setup_logging

logger = setup_logging()

def build_vector_index():
    logger.info("Starting document ingestion process...")
    
    # 1. Load documents using our pipeline
    pipeline = DocumentIngestionPipeline()
    documents = pipeline.load_documents()
    
    if not documents:
        logger.error("No documents found to ingest. Please add files to data/knowledge_base/")
        return
    
    # 2. Configure text splitter (Task 03)
    logger.info(f"Splitting documents with chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Total chunks generated: {len(chunks)}")
    
    # 3. Initialize Embeddings and Vector Store (Task 04 & 05)
    logger.info(f"Initializing embeddings model: {settings.embedding_model_name}")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model_name,
        openai_api_key=settings.openai_api_key
    )
    
    logger.info(f"Persisting vector store to {settings.vector_store_path}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path
    )
    
    logger.info("Vector index built and persisted successfully!")

if __name__ == "__main__":
    build_vector_index()