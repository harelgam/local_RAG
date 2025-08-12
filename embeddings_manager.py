import os
from typing import List, Optional, Tuple
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
        """Create a new vectorstore or load existing one with incremental updates."""
        persist_directory = Config.CHROMA_PERSIST_DIRECTORY

        # Check if vectorstore already exists
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )

            # If new documents provided, add them incrementally
            if documents:
                self.add_documents_incremental(documents)
        else:
            # Create new vectorstore
            if documents:
                print("Creating new vectorstore...")
                # Create empty vectorstore first
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )

                # Now add documents with explicit IDs
                ids = []
                for doc in documents:
                    chunk_id = doc.metadata.get('chunk_id')
                    if chunk_id:
                        ids.append(chunk_id)
                    else:
                        # Generate a simple ID if missing
                        ids.append(f"doc_{len(ids)}")

                # Add documents with IDs
                self.vectorstore.add_documents(
                    documents=documents,
                    ids=ids
                )
                print(f"Added {len(documents)} documents to new vectorstore")
            else:
                print("No documents to process and no existing vectorstore found")
                return None

        return self.vectorstore

    def add_documents_incremental(self, documents: List[Document]) -> int:
        """Add only new documents to the vectorstore (deduplication based on chunk_id)."""
        if not self.vectorstore:
            print("No vectorstore initialized")
            return 0

        # Get existing chunk IDs
        try:
            existing_data = self.vectorstore.get()
            existing_ids = set(existing_data['ids']) if existing_data['ids'] else set()
            print(f"Found {len(existing_ids)} existing chunks in vectorstore")

            # Debug: print existing IDs if not too many
            if len(existing_ids) <= 10:
                print(f"Existing IDs: {existing_ids}")
        except Exception as e:
            print(f"Could not retrieve existing IDs: {e}")
            existing_ids = set()

        # Filter out documents that already exist
        new_documents = []
        skipped_documents = []

        for doc in documents:
            chunk_id = doc.metadata.get('chunk_id')

            if not chunk_id:
                print(f"Warning: Document missing chunk_id, skipping...")
                continue

            print(f"Checking chunk_id: {chunk_id}")
            if chunk_id not in existing_ids:
                new_documents.append(doc)
                print(f"  -> New document (will add)")
            else:
                skipped_documents.append(doc)
                print(f"  -> Duplicate (will skip)")

        # Add new documents
        added_count = 0
        if new_documents:
            print(f"Adding {len(new_documents)} new chunks to vectorstore...")

            # Prepare IDs for Chroma
            ids = [doc.metadata['chunk_id'] for doc in new_documents]

            # Add documents with their IDs
            self.vectorstore.add_documents(
                documents=new_documents,
                ids=ids
            )
            added_count = len(new_documents)
            print(f"✓ Successfully added {added_count} new chunks")
        else:
            print("No new documents to add (all chunks already exist)")

        if skipped_documents:
            print(f"Skipped {len(skipped_documents)} duplicate chunks")

        return added_count

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents (backward compatibility)."""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4,
                                                score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores and filtering.
        Returns list of (document, score) tuples where score > threshold.
        """
        if not self.vectorstore:
            return []

        # Get results with scores
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

        # Filter by threshold
        filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

        if not filtered_results and results:
            print(f"Warning: No results met threshold {score_threshold}. Best score was {results[0][1]:.3f}")

        return filtered_results

    def delete_documents(self, chunk_ids: List[str]) -> bool:
        """Delete specific documents by their chunk IDs."""
        if not self.vectorstore:
            return False

        try:
            self.vectorstore.delete(ids=chunk_ids)
            print(f"Deleted {len(chunk_ids)} documents")
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def update_document(self, chunk_id: str, new_document: Document) -> bool:
        """Update a specific document (delete and re-add)."""
        if not self.vectorstore:
            return False

        try:
            # Delete old version
            self.vectorstore.delete(ids=[chunk_id])

            # Add new version with same ID
            new_document.metadata['chunk_id'] = chunk_id
            self.vectorstore.add_documents(
                documents=[new_document],
                ids=[chunk_id]
            )
            print(f"Updated document with ID: {chunk_id}")
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False

    def get_vectorstore_stats(self) -> dict:
        """Get statistics about the vectorstore."""
        if not self.vectorstore:
            return {"status": "not_initialized"}

        try:
            data = self.vectorstore.get()
            unique_sources = set()

            if data['metadatas']:
                for metadata in data['metadatas']:
                    if metadata and 'source' in metadata:
                        unique_sources.add(metadata['source'])

            return {
                "status": "initialized",
                "total_chunks": len(data['ids']) if data['ids'] else 0,
                "unique_sources": len(unique_sources),
                "sources": list(unique_sources)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}