"""
ChromaDB client setup for the chatbot.
This handles the connection to ChromaDB and provides the base client configuration.
"""

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging

from config import (
    EMBEDDINGS_DIR, DEFAULT_EMBEDDING_FUNCTION
)

logger = logging.getLogger(__name__)

class ChromaClient:
    """
    A wrapper around ChromaDB client to handle connection and configuration.
    """
    
    def __init__(
        self,
        persist_directory: str = EMBEDDINGS_DIR,
        embedding_function_name: str = DEFAULT_EMBEDDING_FUNCTION,
    ):
        """
        Initialize the ChromaDB client.
        
        Args:
            persist_directory: Directory to persist the ChromaDB data
            embedding_function_name: Name of the embedding function to use
        """
        self.persist_directory = persist_directory
        
        # persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # chromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # embedding function setup
        if embedding_function_name == "openai":
            if not (key := os.environ.get("OPENAI_API_KEY")):
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=key,
                model_name="text-embedding-3-small"
            )
        elif embedding_function_name == "huggingface":
            if not (key := os.environ.get("HF_API_KEY")):
                raise ValueError("HuggingFace API key is required for HuggingFace embeddings")
            
            self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=key,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        elif embedding_function_name == "default":
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            raise ValueError(f"Unsupported embedding function: {embedding_function_name}")
        
        logger.info(f"ChromaDB client initialized")

if __name__ == "__main__":
    pass