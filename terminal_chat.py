#!/usr/bin/env python3
import sys
from colorama import init, Fore, Style
from document_loader import DocumentLoader
from embeddings_manager import EmbeddingsManager
from chatbot import RAGChatbot

# Initialize colorama for Windows compatibility
init()


def print_welcome():
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.CYAN}  Company RAG Chatbot - Terminal Interface")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Type 'exit' to quit, 'clear' to clear conversation history")
    print(f"Type 'reload' to reload documents from the data folder{Style.RESET_ALL}\n")


def print_response(response: str):
    print(f"\n{Fore.GREEN}Assistant:{Style.RESET_ALL} {response}\n")


def main():
    print_welcome()

    # Initialize components
    print(f"{Fore.YELLOW}Initializing system...{Style.RESET_ALL}")

    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()

    # Create embeddings and vectorstore
    embeddings_manager = EmbeddingsManager()
    vectorstore = embeddings_manager.create_or_load_vectorstore(documents)

    if not vectorstore:
        print(
            f"{Fore.RED}Error: No documents found. Please add PDF or TXT files to the 'data' folder.{Style.RESET_ALL}")
        sys.exit(1)

    # Initialize chatbot
    chatbot = RAGChatbot(embeddings_manager)

    print(f"{Fore.GREEN}Ready! Start chatting...{Style.RESET_ALL}\n")

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Fore.BLUE}You:{Style.RESET_ALL} ").strip()

            # Handle special commands
            if user_input.lower() == 'exit':
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

            elif user_input.lower() == 'clear':
                chatbot.clear_memory()
                print(f"{Fore.YELLOW}Conversation history cleared.{Style.RESET_ALL}\n")
                continue

            elif user_input.lower() == 'reload':
                print(f"{Fore.YELLOW}Reloading documents...{Style.RESET_ALL}")
                documents = loader.load_documents()
                vectorstore = embeddings_manager.create_or_load_vectorstore(documents)
                print(f"{Fore.GREEN}Documents reloaded successfully!{Style.RESET_ALL}\n")
                continue

            elif not user_input:
                continue

            # Generate and display response
            print(f"{Fore.YELLOW}Thinking...{Style.RESET_ALL}", end='', flush=True)
            response = chatbot.generate_response(user_input)
            print('\r' + ' ' * 20 + '\r', end='')  # Clear "Thinking..." message
            print_response(response)

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Interrupted. Type 'exit' to quit.{Style.RESET_ALL}\n")
            continue
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")
            continue


if __name__ == "__main__":
    main()