# Enhanced config.py with new parameters for improved RAG

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Ollama settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectorstore")

    # Document processing
    DATA_DIRECTORY = "./data"
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # Chatbot settings
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 300))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

    # New settings for enhanced features
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.5))  # Minimum similarity score
    MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", 3))  # Number of query variations
    K_PER_QUERY = int(os.getenv("K_PER_QUERY", 3))  # Documents per query variation