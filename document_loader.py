import os
import hashlib
from datetime import datetime
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config


class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def generate_chunk_id(self, source: str, chunk_index: int, chunk_content: str) -> str:
        """Generate a unique ID for each chunk."""
        # Create a hash of the content for deduplication
        content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
        return f"{source}:chunk_{chunk_index}:{content_hash}"

    def load_documents(self) -> List[Document]:
        """Load all documents from the data directory with chunk IDs."""
        documents = []
        data_dir = Config.DATA_DIRECTORY

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
            return documents

        # Process all files in the data directory
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)

            try:
                file_docs = []

                if filename.endswith('.txt'):
                    print(f"Loading text file: {filename}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    file_docs = loader.load()

                elif filename.endswith('.pdf'):
                    print(f"Loading PDF file: {filename}")
                    loader = PyPDFLoader(file_path)
                    file_docs = loader.load()

                # Add metadata to each document before splitting
                for doc in file_docs:
                    doc.metadata['source'] = filename
                    doc.metadata['load_timestamp'] = datetime.now().isoformat()

                documents.extend(file_docs)

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

        # Split documents into chunks with IDs
        if documents:
            print(f"Loaded {len(documents)} documents. Splitting into chunks...")
            chunks = self.split_documents_with_ids(documents)
            print(f"Created {len(chunks)} chunks with unique IDs")
            return chunks
        else:
            print("No documents found in the data directory")
            return []

    def split_documents_with_ids(self, documents: List[Document]) -> List[Document]:
        """Split documents and assign unique IDs to each chunk."""
        all_chunks = []

        for doc in documents:
            # Split the document
            chunks = self.text_splitter.split_text(doc.page_content)
            source = doc.metadata.get('source', 'unknown')

            # Create Document objects with chunk IDs
            for i, chunk_text in enumerate(chunks):
                chunk_id = self.generate_chunk_id(source, i, chunk_text)

                # Create new document with chunk metadata
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,  # Preserve original metadata
                        'chunk_id': chunk_id,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk_text)
                    }
                )
                all_chunks.append(chunk_doc)

        return all_chunks

    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document with chunk IDs."""
        filename = os.path.basename(file_path)
        documents = []

        try:
            if filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata['source'] = filename
                doc.metadata['load_timestamp'] = datetime.now().isoformat()

            # Split with IDs
            return self.split_documents_with_ids(documents)

        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return []