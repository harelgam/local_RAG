import os
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import Config


class EmbeddingsManager:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=Config.OLLAMA_EMBEDDING_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )
        self.vectorstore = None

    def create_or_load_vectorstore(self, documents: Optional[List[Document]] = None):
        """Create a new vectorstore or load existing one."""
        persist_directory = Config.CHROMA_PERSIST_DIRECTORY

        # Check if vectorstore already exists
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )

            # If new documents provided, add them
            if documents:
                print("Adding new documents to existing vectorstore...")
                self.vectorstore.add_documents(documents)
        else:
            # Create new vectorstore
            if documents:
                print("Creating new vectorstore...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
            else:
                print("No documents to process and no existing vectorstore found")
                return None

        return self.vectorstore

    def similarity_search(self, query: str, k: int = 4):
        """Search for similar documents."""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []