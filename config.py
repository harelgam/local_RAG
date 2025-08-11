# Improved config.py - Better system prompts

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
    TEMPERATURE = 0.3
    MAX_TOKENS = 300
    TOP_K_RESULTS = 5
