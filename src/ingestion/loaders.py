import os
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)

logger = logging.getLogger("SupportPearlz")

class DocumentIngestionPipeline:
    def __init__(self, kb_path: str = "data/knowledge_base"):
        self.kb_path = kb_path

    def load_documents(self) -> List[Document]:
        """
        Recursively walks the knowledge base directory, dispatches loaders 
        based on file extension, attaches required metadata, and handles errors per file.
        """
        if not os.path.exists(self.kb_path):
            logger.error(f"Knowledge base directory not found at {self.kb_path}")
            raise FileNotFoundError(f"Directory not found: {self.kb_path}")

        loaded_documents: List[Document] = []
        stats = {"found": 0, "loaded": 0, "skipped": 0}

        for root, _, files in os.walk(self.kb_path):
            for file in files:
                file_path = os.path.join(root, file)
                stats["found"] += 1
                file_ext = os.path.splitext(file)[1].lower()

                try:
                    docs = self._dispatch_loader(file_path, file_ext)
                    if docs:
                        for doc in docs:
                            # Attach mandatory metadata contract
                            doc.metadata = self._enrich_metadata(file, file_ext, doc.metadata)
                        loaded_documents.extend(docs)
                        stats["loaded"] += 1
                        logger.info(f"Successfully loaded: {file}")
                    else:
                        logger.warning(f"File yielded no content: {file}")
                        stats["skipped"] += 1
                except Exception as e:
                    stats["skipped"] += 1
                    logger.warning(f"Skipping file due to error {file}: {str(e)}")

        logger.info(f"Ingestion summary -> Found: {stats['found']} | Loaded: {stats['loaded']} | Skipped: {stats['skipped']}")
        return loaded_documents

    def _dispatch_loader(self, file_path: str, file_ext: str) -> List[Document]:
        """Selects the correct LangChain loader based on file extension."""
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()
        elif file_ext in [".txt", ".log"]:
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        elif file_ext in [".csv"]:
            loader = CSVLoader(file_path, encoding="utf-8")
            return loader.load()
        else:
            logger.warning(f"Unsupported file extension '{file_ext}' for file: {file_path}")
            return []

    def _enrich_metadata(self, file_name: str, file_ext: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Attaches standardized metadata required for attribution and filtering[span_3](start_span)[span_3](end_span)."""
        doc_type = "general"
        lower_name = file_name.lower()
        
        if "manual" in lower_name:
            doc_type = "manual"
        elif "policy" in lower_name or "agreement" in lower_name:
            doc_type = "policy"
        elif "faq" in lower_name:
            doc_type = "faq"
        elif "pricing" in lower_name or "guide" in lower_name:
            doc_type = "pricing" if "pricing" in lower_name else "guide"

        enriched = {
            "source": file_name,
            "doc_type": doc_type,
            "version": "1.0", # Can be parsed or updated dynamically if tracked
            **existing_metadata
        }
        return enriched

if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    docs = pipeline.load_documents()
    print(f"Total raw chunks/pages loaded across documents: {len(docs)}")