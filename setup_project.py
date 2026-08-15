import os

def create_project_structure():
    # Define directories to create
    directories = [
        "data/knowledge_base",
        "data/vector_store",
        "src/ingestion",
        "src/retrieval",
        "src/chains",
        "src/utils",
        "evaluation/results",
        "docs/screenshots",
        "tests"
    ]

    # Define file templates and their default contents
    files = {
        "requirements.txt": """langchain==0.1.20
langchain-core==0.1.52
langchain-community==0.0.38
langchain-openai==0.1.7
chromadb==0.5.0
pydantic==2.7.1
python-dotenv==1.0.1
tiktoken==0.7.0
rich==13.7.1
typer==0.12.3
""",
        ".env.example": """# LLM & Embedding Configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
TEMPERATURE=0.0

# Vector Store & Paths
VECTOR_STORE_PATH=data/vector_store
COLLECTION_NAME=supportpearlz_kb

# Chunking & Retrieval Tunables
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_K=4
SCORE_THRESHOLD=0.75

# Logging
LOG_LEVEL=INFO
""",
        ".gitignore": """# Environment variables
.env

# Vector Database storage
data/vector_store/

# Python cache and binaries
__pycache__/
*.pyc
*.pyo
*.pyd
.DS_Store

# Evaluation results cache
evaluation/results/*.json
""",
        "src/utils/logging_setup.py": """import logging
import os
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    os.makedirs("data", exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/supportpearlz.log", encoding="utf-8")
        ]
    )
    logger = logging.getLogger("SupportPearlz")
    logger.info("Logging initialized successfully.")
    return logger
""",
        "src/config.py": """import os
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
"""
    }

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")

    # Create files if they don't already exist
    for filepath, content in files.items():
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"File created: {filepath}")
        else:
            print(f"File already exists, skipped: {filepath}")

    print("\nProject structure setup completed successfully!")

if __name__ == "__main__":
    create_project_structure()