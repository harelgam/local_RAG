import streamlit as st
from document_loader import DocumentLoader
from embeddings_manager import EmbeddingsManager
from chatbot import RAGChatbot
import time

# Page configuration
st.set_page_config(
    page_title="Company Support Chat",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: start;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .message-content {
        flex: 1;
        padding: 0 1rem;
    }
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.chatbot = None
    st.session_state.embeddings_manager = None
    # user_input value is stored in session_state via text_input key below

def initialize_system():
    """Initialize the RAG system."""
    with st.spinner("🔄 Loading company knowledge base..."):
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_documents()

        # Create embeddings and vectorstore
        embeddings_manager = EmbeddingsManager()
        vectorstore = embeddings_manager.create_or_load_vectorstore(documents)

        if not vectorstore:
            st.error("⚠️ No documents found. Please add PDF or TXT files to the 'data' folder.")
            return False

        # Initialize chatbot
        st.session_state.embeddings_manager = embeddings_manager
        st.session_state.chatbot = RAGChatbot(embeddings_manager)
        st.session_state.initialized = True

        return True


def display_message(role, content, avatar):
    """Display a chat message."""
    message_class = "user-message" if role == "user" else "assistant-message"

    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="message-avatar">{avatar}</div>
        <div class="message-content">
            <strong>{'You' if role == 'user' else 'Support Agent'}:</strong><br>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def send_message():
    """
    Callback used for both pressing Enter in the text_input (on_change)
    and clicking the Send button (on_click).
    """
    user_input = st.session_state.get("user_input", "")
    if not isinstance(user_input, str):
        return
    user_input = user_input.strip()
    if not user_input:
        # nothing to send
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate assistant response (show spinner while generating)
    if not st.session_state.get("chatbot"):
        # safety: ensure system is initialized
        initialized_ok = initialize_system()
        if not initialized_ok:
            # clear the input so user can try again
            st.session_state.user_input = ""
            return

    with st.spinner("Thinking..."):
        response = st.session_state.chatbot.generate_response(user_input)

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear the input box
    st.session_state.user_input = ""

    # Streamlit will rerun automatically after callback


# Main app
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("💬 Company Support Chat")
        st.caption("Hi! I'm here to help with any questions about our company and services.")

    with col2:
        # Reload button removed by request
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_memory()
            st.experimental_rerun()

    st.divider()

    # Initialize system if needed
    if not st.session_state.initialized:
        if not initialize_system():
            st.stop()

    # Chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_message("user", message["content"], "👤")
            else:
                display_message("assistant", message["content"], "🤖")

    # Input area
    with st.container():
        col1, col2 = st.columns([5, 1])

        with col1:
            # important: key="user_input" makes the value persist in session_state
            # on_change=send_message allows Enter to submit the text_input
            user_input = st.text_input(
                "Type your message...",
                key="user_input",
                label_visibility="collapsed",
                placeholder="Ask me anything about our company...",
                on_change=send_message
            )

        with col2:
            # The Send button now uses on_click to call the same send_message() callback
            st.button("Send 📤", use_container_width=True, on_click=send_message)

    # No manual handling block needed; send_message handles message flow.


if __name__ == "__main__":
    main()
