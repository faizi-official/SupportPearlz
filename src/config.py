import os
from dotenv import load_dotenv
from src.utils.logging_setup import setup_logging

load_dotenv()
logger = setup_logging()

class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
        self.collection_name = os.getenv("COLLECTION_NAME", "supportpearlz_kb")
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "4"))
        self.score_threshold = float(os.getenv("SCORE_THRESHOLD", "0.75"))
        
        self._validate()

    def _validate(self):
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            raise ValueError(f"Startup failed: Missing required environment variables: {', '.join(missing)}")
        logger.info("Configuration validated successfully.")

settings = Settings()
