#!/usr/bin/env python3
"""
Example usage of the RAG system with document category selection
"""

import os
from pathlib import Path
from src.rag.rag import QueryController, MessageType
from src.rag.vector_db_providers.init_vector_db import init_vector_db
from src.utils.manifest_update_handler import update_manifest

def get_document_categories(docs_path):
    """Get available document categories from the documents directory"""
    try:
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            print(f"Warning: Documents path '{docs_path}' does not exist.")
            return []
        
        # Get only directories (not files) from the documents path
        categories = [d.name for d in docs_dir.iterdir() if d.is_dir()]
        return sorted(categories)
    except Exception as e:
        print(f"Error reading document categories: {e}")
        return []

def select_category(categories):
    """Present categories to user and get selection"""
    if not categories:
        print("No document categories found.")
        return None
    
    print("\nAvailable document categories:")
    print("-" * 40)
    
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    print(f"{len(categories) + 1}. All categories")
    print("-" * 40)
    
    while True:
        try:
            choice = input(f"\nSelect a category (1-{len(categories) + 1}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(categories):
                    selected_category = categories[choice_num - 1]
                    print(f"Selected category: {selected_category}")
                    return selected_category
                elif choice_num == len(categories) + 1:
                    print("Selected: All categories")
                    return None  # None indicates all categories
                else:
                    print(f"Please enter a number between 1 and {len(categories) + 1}")
            else:
                print("Please enter a valid number")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return False
        except Exception as e:
            print(f"Invalid input: {e}")

def main():
    """Run user queries and process them using RAG"""
    
    # Define the documents path (you can make this configurable)
    docs_path = '/Users/rodrigocampos/knowledge-base-test/documents'
    
    print(f"Looking for documents in: {docs_path}")
    
    # Get available document categories
    categories = get_document_categories(docs_path)
    
    if not categories:
        print("No document categories found. Proceeding with default configuration...")
        selected_category = None
    else:
        # Let user select category
        selected_category = select_category(categories)
        
        if selected_category is False:  # User pressed Ctrl+C
            return
    
    # Initialize the RAG system
    print("\nInitializing RAG system...")
    vector_db_provider = init_vector_db()
    rag = QueryController(vector_db_provider)
    
    if selected_category:
        print(f"RAG system configured for category: {selected_category}")
        # TODO: Configure RAG to focus on selected_category if your system supports it
        # rag.set_category_filter(selected_category)
    else:
        print("RAG system configured for all categories")

    print("\nEnter your questions:")
    print("(Type 'change-category' to select a different category)")
    
    while True:
        message = input("\nYour question (or 'quit' to exit): ").strip()
        
        if message.lower() in ['quit', 'exit', 'q']:
            break
        
        if message.lower() == 'change-category':
            # Allow user to change category during the session
            new_category = select_category(categories)
            if new_category is False:
                break
            selected_category = new_category
            if selected_category:
                print(f"Switched to category: {selected_category}")
                # TODO: Reconfigure RAG for new category
                # rag.set_category_filter(selected_category)
            else:
                print("Switched to all categories")
                # TODO: Remove category filter
                # rag.clear_category_filter()
            continue
        
        if message:
            try:
                message_with_type = rag.feedback_or_query(message)
                print(f"MESSAGE TYPE: {message_with_type['type']}")
                
                if message_with_type["type"] == MessageType.QUERY:
                    answer = rag.query_with_rag(selected_category, message)
                    print(f"Answer: {answer}")
                elif message_with_type["type"] == MessageType.FEEDBACK:
                    update_manifest(rag, message)
                else:
                    print(MessageType.FEEDBACK)
                    answer = rag.query_with_rag(selected_category, message)
                    print(f"Answer: {answer}")
                    
            except Exception as e:
                print(f"Error processing query: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()