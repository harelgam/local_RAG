#!/usr/bin/env python3
"""
test_multidoc_queries.py - Tests the chatbot's ability to combine information from multiple documents
"""

from document_loader import DocumentLoader
from embeddings_manager import EmbeddingsManager
from chatbot import RAGChatbot
from colorama import init, Fore, Style
import json

# Initialize colorama
init()


def test_multi_document_queries():
    """Test queries that require information from multiple documents."""

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("TESTING MULTI-DOCUMENT QUERY HANDLING")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    # Initialize the chatbot
    print(f"{Fore.YELLOW}Initializing chatbot...{Style.RESET_ALL}")
    loader = DocumentLoader()
    documents = loader.load_documents()

    if not documents:
        print(f"{Fore.RED}No documents found! Run setup_examples.py first.{Style.RESET_ALL}")
        return

    embeddings_manager = EmbeddingsManager()
    vectorstore = embeddings_manager.create_or_load_vectorstore(documents)
    chatbot = RAGChatbot(embeddings_manager)

    # Test queries that REQUIRE multiple documents
    test_queries = [
        {
            'query': "I'm a nonprofit with 25 employees. What plan should I choose and what discount do I get?",
            'required_files': ['pricing_plans.txt', 'faq.txt'],
            'expected_info': [
                'Professional plan (for 25 users)',
                '20% nonprofit discount',
                '$999/month regular price',
                'Discounted price around $799/month'
            ]
        },
        {
            'query': "I can't login and need urgent help. What's your phone number and response time for critical issues?",
            'required_files': ['troubleshooting.txt', 'support_guide.txt', 'company_overview.txt'],
            'expected_info': [
                'Login troubleshooting steps',
                'Phone: 1-800-FLOW-TECH',
                'Critical response times by plan',
                'Clear cache/cookies solution'
            ]
        },
        {
            'query': "Tell me about CloudFlow Pro pricing and what security certifications you have",
            'required_files': ['product_features.txt', 'pricing_plans.txt', 'company_overview.txt'],
            'expected_info': [
                'CloudFlow Pro features',
                'Plan pricing ($299/$999/custom)',
                'SOC 2 Type II certified',
                'ISO 27001 certified'
            ]
        },
        {
            'query': "We need HIPAA compliance, API access for 1 million calls per month, and 24/7 support. What do you recommend?",
            'required_files': ['faq.txt', 'pricing_plans.txt', 'support_guide.txt', 'company_overview.txt'],
            'expected_info': [
                'Enterprise plan recommendation',
                'HIPAA/BAA available',
                '24/7 support with Enterprise',
                'Unlimited or high API limits'
            ]
        },
        {
            'query': "I'm getting error 429 from the API and I'm on the Professional plan. Also, how do I escalate if this doesn't get fixed quickly?",
            'required_files': ['troubleshooting.txt', 'support_guide.txt', 'pricing_plans.txt'],
            'expected_info': [
                'Error 429 = rate limit exceeded',
                'Professional plan: 10,000 calls/day',
                'Escalation process',
                'Wait 60 seconds solution'
            ]
        }
    ]

    # Test each query
    for i, test in enumerate(test_queries, 1):
        print(f"\n{Fore.BLUE}Test {i}: Multi-Document Query{Style.RESET_ALL}")
        print(f"Query: {test['query']}")
        print(f"Should combine info from: {', '.join(test['required_files'])}")

        # Get response
        response = chatbot.generate_response(test['query'])

        print(f"\n{Fore.GREEN}Bot Response:{Style.RESET_ALL}")
        print(response[:500] + "..." if len(response) > 500 else response)

        # Check if response contains expected information
        print(f"\n{Fore.YELLOW}Information Coverage Check:{Style.RESET_ALL}")
        for info in test['expected_info']:
            # Simple check - in production you'd want more sophisticated validation
            found = any(keyword.lower() in response.lower()
                        for keyword in info.split()
                        if len(keyword) > 4)
            status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if found else f"{Fore.RED}✗{Style.RESET_ALL}"
            print(f"{status} {info}")

        print("-" * 60)

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("ADVANCED TESTING: DOCUMENT SOURCE TRACKING")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    # Show which documents were actually retrieved
    test_query = "I need pricing for 30 users with HIPAA compliance and 24/7 support"
    print(f"Query: {test_query}\n")

    # Get the actual documents retrieved
    relevant_docs = embeddings_manager.similarity_search(test_query, k=4)

    print(f"{Fore.YELLOW}Documents Retrieved:{Style.RESET_ALL}")
    sources = {}
    for idx, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get('source', 'Unknown').split('\\')[-1].split('/')[-1]
        chunk = doc.page_content[:100] + "..."

        if source not in sources:
            sources[source] = []
        sources[source].append(idx)

        print(f"\n{Fore.BLUE}Chunk {idx} from {source}:{Style.RESET_ALL}")
        print(chunk)

    print(f"\n{Fore.GREEN}Summary:{Style.RESET_ALL}")
    print(f"Retrieved chunks from {len(sources)} different files:")
    for source, chunks in sources.items():
        print(f"  - {source}: chunks {chunks}")

    if len(sources) >= 2:
        print(f"\n{Fore.GREEN}✅ SUCCESS: Bot is combining information from multiple files!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}⚠️ WARNING: Bot might not be effectively combining multiple sources{Style.RESET_ALL}")


def test_improved_retrieval():
    """Test an improved version with explicit multi-doc handling."""

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("SUGGESTIONS FOR BETTER MULTI-DOCUMENT HANDLING")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    suggestions = [
        {
            'title': '1. Increase K value for complex queries',
            'code': '''# In chatbot.py
def generate_response(self, query: str) -> str:
    # Detect complex queries that need more context
    complex_keywords = ['and', 'also', 'plus', 'with', 'including']
    k_value = 6 if any(kw in query.lower() for kw in complex_keywords) else 3

    relevant_docs = self.embeddings_manager.similarity_search(query, k=k_value)'''
        },
        {
            'title': '2. Use query expansion',
            'code': '''# Expand query to capture related concepts
def expand_query(self, query: str) -> str:
    expansions = {
        'HIPAA': 'HIPAA compliance healthcare medical BAA security',
        'pricing': 'pricing cost price plan subscription fee',
        'support': 'support help assistance contact phone email'
    }

    expanded = query
    for term, expansion in expansions.items():
        if term.lower() in query.lower():
            expanded += f" {expansion}"
    return expanded'''
        },
        {
            'title': '3. Ensure diverse source retrieval',
            'code': '''# Get at least one chunk from each relevant document type
def get_diverse_documents(self, query: str) -> List:
    all_docs = []
    doc_types = ['pricing', 'support', 'features', 'faq', 'troubleshooting']

    for doc_type in doc_types:
        type_query = f"{query} {doc_type}"
        docs = self.similarity_search(type_query, k=1)
        all_docs.extend(docs)

    return all_docs[:6]  # Return top 6 diverse chunks'''
        },
        {
            'title': '4. Add metadata filtering',
            'code': '''# Filter by document type when needed
def search_with_filter(self, query: str, doc_filter: str = None):
    if doc_filter:
        # ChromaDB supports metadata filtering
        return self.vectorstore.similarity_search(
            query, 
            k=4,
            filter={"source": {"$contains": doc_filter}}
        )
    return self.vectorstore.similarity_search(query, k=4)'''
        }
    ]

    for suggestion in suggestions:
        print(f"{Fore.YELLOW}{suggestion['title']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{suggestion['code']}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    print("""
This script tests whether your chatbot can effectively combine 
information from multiple documents to answer complex queries.
    """)

    test_multi_document_queries()
    test_improved_retrieval()

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("TESTING COMPLETE")
    print(f"{'=' * 60}{Style.RESET_ALL}")
    print("""
If the bot isn't combining documents well, consider:
1. Increasing k value (number of chunks retrieved)
2. Improving chunk size and overlap
3. Using query expansion techniques
4. Implementing diverse source retrieval
""")