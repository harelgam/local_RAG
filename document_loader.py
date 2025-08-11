import os
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

    def load_documents(self) -> List[Document]:
        """Load all documents from the data directory."""
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
                if filename.endswith('.txt'):
                    print(f"Loading text file: {filename}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())

                elif filename.endswith('.pdf'):
                    print(f"Loading PDF file: {filename}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

        # Split documents into chunks
        if documents:
            print(f"Loaded {len(documents)} documents. Splitting into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            return chunks
        else:
            print("No documents found in the data directory")
            return []