from typing import List, Dict, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage, Document
from langchain.memory import ConversationBufferMemory
from config import Config
from embeddings_manager import EmbeddingsManager


class RAGChatbot:
    def __init__(self, embeddings_manager: EmbeddingsManager):
        self.embeddings_manager = embeddings_manager
        self.llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=Config.TEMPERATURE,
            num_predict=Config.MAX_TOKENS
        )
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.conversation_history = []
        self.has_introduced = False  # Track if we've introduced ourselves

    def generate_multiple_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple versions of a query for better retrieval."""
        prompt = f"""You are a search query generator. Your task is to create {num_queries} alternative phrasings of a question.

Original question: {original_query}

Generate {num_queries} alternative ways to ask this same question. Each should be a complete question that means the same thing but uses different words.

Alternative versions (one per line):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            # Parse the response to get individual queries
            alternative_queries = response.content.strip().split('\n')
            # Clean up the queries
            alternative_queries = [q.strip() for q in alternative_queries if q.strip()]

            # Include original query and return top num_queries
            all_queries = [original_query] + alternative_queries
            return all_queries[:num_queries + 1]
        except Exception as e:
            print(f"Error generating multiple queries: {e}")
            # Fallback to just the original query
            return [original_query]

    def retrieve_with_multi_query(self, query: str, k_per_query: int = 3) -> Tuple[List[Document], List[str]]:
        """
        Retrieve documents using multiple query variations.
        Returns (documents, sources)
        """
        # Generate multiple queries
        queries = self.generate_multiple_queries(query)
        print(f"Generated {len(queries)} query variations")

        # Retrieve for each query with relevance filtering
        all_docs = []
        seen_content = set()  # For deduplication

        for q in queries:
            results = self.embeddings_manager.similarity_search_with_relevance_scores(
                q,
                k=k_per_query,
                score_threshold=Config.RELEVANCE_THRESHOLD
            )

            for doc, score in results:
                # Deduplicate based on content
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append((doc, score))

        # If no docs found with threshold, try with lower threshold as fallback
        if not all_docs and Config.RELEVANCE_THRESHOLD > 0.3:
            print(f"No docs found with threshold {Config.RELEVANCE_THRESHOLD}, trying with 0.3")
            for q in queries[:1]:  # Just try with original query
                results = self.embeddings_manager.similarity_search_with_relevance_scores(
                    q,
                    k=k_per_query,
                    score_threshold=0.3
                )
                for doc, score in results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append((doc, score))

        # Sort by score and take top k
        all_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = all_docs[:Config.TOP_K_RESULTS]

        # Extract documents and sources
        documents = [doc for doc, _ in top_docs]
        sources = []
        for doc, score in top_docs:
            source_info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                'score': f"{score:.3f}"
            }
            sources.append(source_info)

        return documents, sources

    def format_context_with_sources(self, documents: List[Document], sources: List[dict]) -> str:
        """Format retrieved documents as context with source tracking."""
        if not documents:
            return ""

        context_parts = []
        for i, (doc, source) in enumerate(zip(documents, sources)):
            context_parts.append(f"[Document {i + 1} - {source['source']} (relevance: {source['score']})]")
            context_parts.append(doc.page_content)
            context_parts.append("")  # Empty line between documents

        return "\n".join(context_parts).strip()

    def generate_response(self, query: str) -> str:
        """Generate a response using multi-query RAG."""
        # Retrieve relevant documents using multi-query
        relevant_docs, sources = self.retrieve_with_multi_query(query)

        # Check if we found relevant documents
        if not relevant_docs:
            no_info_response = "I don't have enough information in my knowledge base to answer that question accurately. Could you please rephrase your question or ask about something else?"

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": no_info_response})

            return no_info_response

        # Format context from retrieved documents
        context = self.format_context_with_sources(relevant_docs, sources)

        # Build conversation history string
        history_str = ""
        if self.conversation_history:
            for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                role = "Customer" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        # Adjust system prompt based on whether we've introduced ourselves
        if not self.has_introduced and len(self.conversation_history) == 0:
            system_prompt = """You are a helpful customer service representative for TechFlow Solutions. 
Your name is Alex. This is the first interaction, so briefly introduce yourself.
Be friendly, professional, and conversational. Keep responses concise and natural.
Use the provided context to answer questions accurately.
Always mention which document your information comes from when answering.
If you don't know something, politely say so."""
            self.has_introduced = True
        else:
            system_prompt = """You are Alex, a customer service representative for TechFlow Solutions.
You've already introduced yourself, so don't do it again.
Be friendly, professional, and conversational. Keep responses concise and natural.
Use the provided context to answer questions accurately.
When providing information, subtly reference the source (e.g., 'According to our documentation...', 'Our policy states...').
If you don't know something, politely say so."""

        # Create the prompt
        prompt = f"""Based on this information:
{context}

Previous conversation:
{history_str}

Customer's question: {query}

Provide a helpful, natural response. Reference the information naturally without listing document numbers."""

        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # Format final response with sources
        final_response = response.content

        # Add source citations at the end
        if sources:
            final_response += "\n\n📚 *Sources used:*\n"
            unique_sources = {}
            for source in sources:
                src_name = source['source']
                if src_name not in unique_sources:
                    unique_sources[src_name] = source['score']

            for src_name, score in unique_sources.items():
                final_response += f"• {src_name}\n"

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.content})

        # Keep only last 10 exchanges to manage context length
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return final_response

    def clear_memory(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.memory.clear()
        self.has_introduced = False  # Reset introduction flag