#!/usr/bin/env python3
"""
Test script to verify all enhanced features are working correctly.
Run this after implementing the changes to ensure everything works.
"""

import os
import gc
import time
import tempfile
from colorama import init, Fore, Style
from document_loader import DocumentLoader
from embeddings_manager import EmbeddingsManager
from chatbot import RAGChatbot

# Initialize colorama
init()


def test_chunk_id_system():
    """Test that chunk IDs are being generated correctly."""
    print(f"\n{Fore.CYAN}=== Testing Chunk ID System ==={Style.RESET_ALL}")

    # Create a temporary test document
    test_content = """This is a test document.

    It has multiple paragraphs to ensure chunking works.

    This is the third paragraph with enough content to potentially create multiple chunks if needed.
    Let's add more text here to make it substantial."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        loader = DocumentLoader()
        chunks = loader.load_single_document(temp_file)

        print(f"Created {len(chunks)} chunks")

        # Check each chunk has required metadata
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get('chunk_id')
            assert chunk_id is not None, f"Chunk {i} missing chunk_id"
            assert 'source' in chunk.metadata, f"Chunk {i} missing source"
            assert 'chunk_index' in chunk.metadata, f"Chunk {i} missing chunk_index"

            print(f"  Chunk {i}: ID = {chunk_id}")

        print(f"{Fore.GREEN}✓ Chunk ID system working correctly{Style.RESET_ALL}")
        return True

    except Exception as e:
        print(f"{Fore.RED}✗ Chunk ID test failed: {e}{Style.RESET_ALL}")
        return False
    finally:
        os.unlink(temp_file)


def test_incremental_updates():
    """Test incremental update functionality."""
    print(f"\n{Fore.CYAN}=== Testing Incremental Updates ==={Style.RESET_ALL}")

    # Create test vectorstore in temp directory
    import tempfile
    import shutil
    from config import Config

    original_persist_dir = Config.CHROMA_PERSIST_DIRECTORY
    temp_dir = tempfile.mkdtemp()
    Config.CHROMA_PERSIST_DIRECTORY = temp_dir

    try:
        # Create initial documents with unique IDs
        from langchain.schema import Document

        docs_batch1 = [
            Document(
                page_content="First document content",
                metadata={'chunk_id': 'doc1_chunk_0', 'source': 'test1.txt'}
            ),
            Document(
                page_content="Second document content",
                metadata={'chunk_id': 'doc1_chunk_1', 'source': 'test1.txt'}
            )
        ]

        # Initialize and add first batch
        em = EmbeddingsManager()
        em.create_or_load_vectorstore(docs_batch1)

        stats1 = em.get_vectorstore_stats()
        print(f"Initial chunks: {stats1['total_chunks']}")
        assert stats1['total_chunks'] == 2, "Should have 2 initial chunks"

        # Debug: Check what IDs were actually stored
        stored_data = em.vectorstore.get()
        print(f"Actually stored IDs after initial creation: {stored_data['ids']}")

        # Add second batch with one duplicate and one new
        docs_batch2 = [
            Document(
                page_content="Second document content",  # Same content, same ID = duplicate
                metadata={'chunk_id': 'doc1_chunk_1', 'source': 'test1.txt'}
            ),
            Document(
                page_content="Third document content",  # New content, new ID
                metadata={'chunk_id': 'doc2_chunk_0', 'source': 'test2.txt'}
            )
        ]

        # This should only add 1 new document (doc2_chunk_0)
        added = em.add_documents_incremental(docs_batch2)
        print(f"Function returned: added {added} new chunks")

        stats2 = em.get_vectorstore_stats()
        print(f"Final chunks: {stats2['total_chunks']}")

        # Debug: Check final IDs
        final_data = em.vectorstore.get()
        print(f"Final stored IDs: {final_data['ids']}")

        # Should have 3 total chunks (2 original + 1 new, duplicate was skipped)
        assert stats2[
                   'total_chunks'] == 3, f"Should have 3 total chunks (2 original + 1 new), got {stats2['total_chunks']}"
        assert added == 1, f"Should have added only 1 new chunk, got {added}"

        print(f"{Fore.GREEN}✓ Incremental updates working correctly{Style.RESET_ALL}")

        # Clean up vectorstore properly
        del em
        gc.collect()
        time.sleep(0.1)  # Give Windows time to release file handles

        return True

    except Exception as e:
        print(f"{Fore.RED}✗ Incremental update test failed: {e}{Style.RESET_ALL}")
        return False
    finally:
        Config.CHROMA_PERSIST_DIRECTORY = original_persist_dir
        # Force cleanup on Windows
        gc.collect()
        time.sleep(0.2)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass  # Ignore cleanup errors on Windows


def test_relevance_filtering():
    """Test relevance score filtering."""
    print(f"\n{Fore.CYAN}=== Testing Relevance Filtering ==={Style.RESET_ALL}")

    # Create test vectorstore
    import tempfile
    import shutil
    from config import Config
    from langchain.schema import Document

    original_persist_dir = Config.CHROMA_PERSIST_DIRECTORY
    temp_dir = tempfile.mkdtemp()
    Config.CHROMA_PERSIST_DIRECTORY = temp_dir

    try:
        # Create test documents with more distinct content
        docs = [
            Document(
                page_content="Python is a high-level programming language known for its simplicity",
                metadata={'chunk_id': 'test:0', 'source': 'test.txt'}
            ),
            Document(
                page_content="Coffee is a popular beverage enjoyed by millions worldwide",
                metadata={'chunk_id': 'test:1', 'source': 'test.txt'}
            ),
            Document(
                page_content="Machine learning algorithms often use Python for implementation",
                metadata={'chunk_id': 'test:2', 'source': 'test.txt'}
            )
        ]

        em = EmbeddingsManager()
        em.create_or_load_vectorstore(docs)

        # Test with moderate threshold
        results_high = em.similarity_search_with_relevance_scores(
            "Python programming language",
            k=3,
            score_threshold=0.6  # Lowered from 0.7
        )

        # Test with low threshold
        results_low = em.similarity_search_with_relevance_scores(
            "Python programming language",
            k=3,
            score_threshold=0.3
        )

        print(f"Moderate threshold (0.6): {len(results_high)} results")
        print(f"Low threshold (0.3): {len(results_low)} results")

        # High threshold should return fewer results
        assert len(results_high) <= len(results_low), "Higher threshold should be more selective"

        # Check scores are above threshold
        for doc, score in results_high:
            assert score >= 0.6, f"Score {score} below threshold 0.6"
            print(f"  Score: {score:.3f} - {doc.page_content[:50]}...")

        print(f"{Fore.GREEN}✓ Relevance filtering working correctly{Style.RESET_ALL}")

        # Clean up
        del em
        gc.collect()
        time.sleep(0.1)

        return True

    except Exception as e:
        print(f"{Fore.RED}✗ Relevance filtering test failed: {e}{Style.RESET_ALL}")
        return False
    finally:
        Config.CHROMA_PERSIST_DIRECTORY = original_persist_dir
        gc.collect()
        time.sleep(0.2)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def test_multi_query_retrieval():
    """Test multi-query generation and retrieval."""
    print(f"\n{Fore.CYAN}=== Testing Multi-Query Retrieval ==={Style.RESET_ALL}")

    import tempfile
    import shutil
    from config import Config
    from langchain.schema import Document

    original_persist_dir = Config.CHROMA_PERSIST_DIRECTORY
    original_threshold = Config.RELEVANCE_THRESHOLD
    temp_dir = tempfile.mkdtemp()
    Config.CHROMA_PERSIST_DIRECTORY = temp_dir
    Config.RELEVANCE_THRESHOLD = 0.5  # Lower threshold for test

    try:
        # Create test documents with clearer content
        docs = [
            Document(
                page_content="Our company refund policy: customers can return products within 30 days for a full refund",
                metadata={'chunk_id': 'policy:0', 'source': 'policy.txt'}
            ),
            Document(
                page_content="To get a refund, customers must provide proof of purchase and return items in original condition",
                metadata={'chunk_id': 'policy:1', 'source': 'policy.txt'}
            ),
            Document(
                page_content="The money back guarantee ensures customer satisfaction with all purchases",
                metadata={'chunk_id': 'policy:2', 'source': 'policy.txt'}
            )
        ]

        em = EmbeddingsManager()
        em.create_or_load_vectorstore(docs)

        chatbot = RAGChatbot(em)

        # Test multi-query generation
        original_query = "What is the refund policy?"
        queries = chatbot.generate_multiple_queries(original_query)

        print(f"Original query: '{original_query}'")
        print(f"Generated {len(queries)} query variations:")
        for i, q in enumerate(queries):
            print(f"  {i + 1}. {q}")

        assert len(queries) >= 2, "Should generate multiple queries"
        assert original_query in queries, "Should include original query"

        # Test retrieval with multi-query using lower threshold
        docs_retrieved, sources = chatbot.retrieve_with_multi_query(original_query, k_per_query=3)

        print(f"\nRetrieved {len(docs_retrieved)} documents")
        for i, (doc, src) in enumerate(zip(docs_retrieved, sources)):
            print(f"  Doc {i + 1}: {src['source']} (score: {src['score']})")

        assert len(docs_retrieved) > 0, "Should retrieve some documents"

        print(f"{Fore.GREEN}✓ Multi-query retrieval working correctly{Style.RESET_ALL}")

        # Clean up
        del chatbot
        del em
        gc.collect()
        time.sleep(0.1)

        return True

    except Exception as e:
        print(f"{Fore.RED}✗ Multi-query retrieval test failed: {e}{Style.RESET_ALL}")
        return False
    finally:
        Config.CHROMA_PERSIST_DIRECTORY = original_persist_dir
        Config.RELEVANCE_THRESHOLD = original_threshold
        gc.collect()
        time.sleep(0.2)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def run_all_tests():
    """Run all tests and report results."""
    print(f"\n{Fore.YELLOW}{'=' * 50}")
    print("Running Enhanced Features Test Suite")
    print(f"{'=' * 50}{Style.RESET_ALL}")

    tests = [
        ("Chunk ID System", test_chunk_id_system),
        ("Incremental Updates", test_incremental_updates),
        ("Relevance Filtering", test_relevance_filtering),
        ("Multi-Query Retrieval", test_multi_query_retrieval)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"{Fore.RED}Test '{name}' crashed: {e}{Style.RESET_ALL}")
            results.append((name, False))

    # Print summary
    print(f"\n{Fore.YELLOW}{'=' * 50}")
    print("Test Summary")
    print(f"{'=' * 50}{Style.RESET_ALL}")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_test in results:
        status = f"{Fore.GREEN}✓ PASSED{Style.RESET_ALL}" if passed_test else f"{Fore.RED}✗ FAILED{Style.RESET_ALL}"
        print(f"{name}: {status}")

    print(f"\n{Fore.CYAN}Results: {passed}/{total} tests passed{Style.RESET_ALL}")

    if passed == total:
        print(f"{Fore.GREEN}🎉 All tests passed! Your enhanced features are working correctly.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️ Some tests failed. Please review the output above.{Style.RESET_ALL}")

    # Final cleanup for Windows
    gc.collect()


if __name__ == "__main__":
    run_all_tests()