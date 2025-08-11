from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
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

    def format_context(self, documents: List) -> str:
        """Format retrieved documents as context WITHOUT document references."""
        if not documents:
            return ""

        # Don't mention documents - just provide the information
        context = ""
        for doc in documents:
            context += f"{doc.page_content}\n\n"

        return context.strip()

    def generate_response(self, query: str) -> str:
        """Generate a response using RAG."""
        # Retrieve relevant documents
        relevant_docs = self.embeddings_manager.similarity_search(
            query,
            k=Config.TOP_K_RESULTS
        )

        # Format context from retrieved documents
        context = self.format_context(relevant_docs)

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
If you don't know something, politely say so."""
            self.has_introduced = True
        else:
            system_prompt = """You are Alex, a customer service representative for TechFlow Solutions.
You've already introduced yourself, so don't do it again.
Be friendly, professional, and conversational. Keep responses concise and natural.
Use the provided context to answer questions accurately.
Never mention document numbers, sources, or that you're looking at documents.
Respond as if you naturally know this information.
If you don't know something, politely say so."""

        # Create the prompt
        if context:
            prompt = f"""Based on this information:
{context}

Previous conversation:
{history_str}

Customer's question: {query}

Provide a helpful, natural response. Don't mention documents or sources."""
        else:
            prompt = f"""Previous conversation:
{history_str}

Customer's question: {query}

Provide a helpful, natural response based on what you know about TechFlow Solutions."""

        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.content})

        # Keep only last 10 exchanges to manage context length
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response.content

    def clear_memory(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.memory.clear()
        self.has_introduced = False  # Reset introduction flag