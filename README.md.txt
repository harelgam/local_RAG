# Company RAG Chatbot

A 100% local, private RAG (Retrieval-Augmented Generation) chatbot for company customer support using Ollama, LangChain, and ChromaDB.

## Features

- 🔒 **100% Local & Private**: All data stays on your machine
- 🧠 **Conversation Memory**: Remembers context within sessions
- 📚 **Document Support**: Works with PDF and TXT files
- 💬 **Natural Conversations**: Responds like a human support agent
- 🖥️ **Dual Interface**: Terminal and web (Streamlit) options
- 🚀 **Fast & Free**: Uses open-source tools

## Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama from https://ollama.com
   
   # Pull required models
   ollama pull llama3.2:3b
   ollama pull mxbai-embed-large