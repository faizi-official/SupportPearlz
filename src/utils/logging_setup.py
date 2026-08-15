import logging
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
