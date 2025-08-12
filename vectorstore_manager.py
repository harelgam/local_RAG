#!/usr/bin/env python3
"""
Vectorstore Management Utility
Provides commands to manage the enhanced vectorstore with incremental updates.
"""

import sys
import argparse
from colorama import init, Fore, Style
from document_loader import DocumentLoader
from embeddings_manager import EmbeddingsManager
import os
import shutil

# Initialize colorama
init()


def print_stats(embeddings_manager: EmbeddingsManager):
    """Print vectorstore statistics."""
    stats = embeddings_manager.get_vectorstore_stats()

    print(f"\n{Fore.CYAN}=== Vectorstore Statistics ==={Style.RESET_ALL}")
    print(f"Status: {stats.get('status', 'unknown')}")

    if stats['status'] == 'initialized':
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        print(f"Unique sources: {stats.get('unique_sources', 0)}")

        if stats.get('sources'):
            print(f"\n{Fore.YELLOW}Sources:{Style.RESET_ALL}")
            for source in stats['sources']:
                print(f"  • {source}")
    elif stats['status'] == 'error':
        print(f"{Fore.RED}Error: {stats.get('error', 'Unknown error')}{Style.RESET_ALL}")


def add_document(file_path: str, embeddings_manager: EmbeddingsManager):
    """Add a single document to the vectorstore."""
    if not os.path.exists(file_path):
        print(f"{Fore.RED}Error: File '{file_path}' not found{Style.RESET_ALL}")
        return

    print(f"{Fore.YELLOW}Loading document: {file_path}{Style.RESET_ALL}")

    loader = DocumentLoader()
    documents = loader.load_single_document(file_path)

    if documents:
        print(f"Document split into {len(documents)} chunks")
        added = embeddings_manager.add_documents_incremental(documents)
        print(f"{Fore.GREEN}✓ Added {added} new chunks to vectorstore{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed to load document{Style.RESET_ALL}")


def rebuild_vectorstore():
    """Rebuild the entire vectorstore from scratch."""
    print(f"{Fore.YELLOW}Rebuilding vectorstore from scratch...{Style.RESET_ALL}")

    # Delete existing vectorstore
    from config import Config
    if os.path.exists(Config.CHROMA_PERSIST_DIRECTORY):
        shutil.rmtree(Config.CHROMA_PERSIST_DIRECTORY)
        print(f"Deleted existing vectorstore at {Config.CHROMA_PERSIST_DIRECTORY}")

    # Load all documents
    loader = DocumentLoader()
    documents = loader.load_documents()

    if documents:
        # Create new vectorstore
        embeddings_manager = EmbeddingsManager()
        embeddings_manager.create_or_load_vectorstore(documents)
        print(f"{Fore.GREEN}✓ Vectorstore rebuilt with {len(documents)} chunks{Style.RESET_ALL}")
        print_stats(embeddings_manager)
    else:
        print(f"{Fore.RED}No documents found to index{Style.RESET_ALL}")


def update_all_documents():
    """Check for new documents and add them incrementally."""
    print(f"{Fore.YELLOW}Checking for new documents...{Style.RESET_ALL}")

    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()

    if not documents:
        print(f"{Fore.YELLOW}No documents found in data directory{Style.RESET_ALL}")
        return

    # Initialize embeddings manager
    embeddings_manager = EmbeddingsManager()
    vectorstore = embeddings_manager.create_or_load_vectorstore(documents)

    if vectorstore:
        print(f"{Fore.GREEN}✓ Update complete{Style.RESET_ALL}")
        print_stats(embeddings_manager)


def test_retrieval(query: str):
    """Test retrieval with a query."""
    embeddings_manager = EmbeddingsManager()
    embeddings_manager.create_or_load_vectorstore()

    print(f"\n{Fore.CYAN}Testing retrieval for: '{query}'{Style.RESET_ALL}")

    # Test with relevance scores
    results = embeddings_manager.similarity_search_with_relevance_scores(
        query,
        k=5,
        score_threshold=0.7
    )

    if results:
        print(f"\n{Fore.GREEN}Found {len(results)} relevant documents:{Style.RESET_ALL}")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{Fore.YELLOW}Document {i} (Score: {score:.3f}):{Style.RESET_ALL}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"Content preview: {doc.page_content[:200]}...")
    else:
        print(f"{Fore.RED}No documents found above relevance threshold{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description='Vectorstore Management Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Stats command
    subparsers.add_parser('stats', help='Show vectorstore statistics')

    # Add document command
    add_parser = subparsers.add_parser('add', help='Add a single document')
    add_parser.add_argument('file', help='Path to the document file')

    # Rebuild command
    subparsers.add_parser('rebuild', help='Rebuild vectorstore from scratch')

    # Update command
    subparsers.add_parser('update', help='Update vectorstore with new documents')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test retrieval with a query')
    test_parser.add_argument('query', help='Query to test')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize embeddings manager for most commands
    if args.command in ['stats', 'add', 'test']:
        embeddings_manager = EmbeddingsManager()
        embeddings_manager.create_or_load_vectorstore()

    # Execute commands
    if args.command == 'stats':
        print_stats(embeddings_manager)

    elif args.command == 'add':
        add_document(args.file, embeddings_manager)

    elif args.command == 'rebuild':
        response = input(f"{Fore.YELLOW}This will delete the existing vectorstore. Continue? (y/n): {Style.RESET_ALL}")
        if response.lower() == 'y':
            rebuild_vectorstore()
        else:
            print("Operation cancelled")

    elif args.command == 'update':
        update_all_documents()

    elif args.command == 'test':
        test_retrieval(args.query)


if __name__ == "__main__":
    main()