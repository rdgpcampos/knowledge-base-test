#!/usr/bin/env python3
"""
Example usage of the RAG system
"""

from src.rag.rag import QueryController, MessageType
from src.rag.vector_db_providers.init_vector_db import init_vector_db
from src.utils.manifest_update_handler import update_manifest

def main():
    # Initialize the RAG system
    vector_db_provider = init_vector_db()
    rag = QueryController(vector_db_provider)

    print("Enter your questions:")
    
    while True:
        message = input("\nYour question (or 'quit' to exit): ")
        
        if message.lower() in ['quit', 'exit', 'q']:
            break
        
        if message.strip():
            message_with_type = rag.feedback_or_query(message)
            print(f"MESSAGE TYPE: {message_with_type['type']}")
            if message_with_type["type"] == MessageType.QUERY:
                answer = rag.query_with_rag(message)
                print(f"Answer: {answer}")
            elif message_with_type["type"] == MessageType.FEEDBACK:
                update_manifest(rag, message)
            else:
                print(MessageType.FEEDBACK)
                answer = rag.query_with_rag(message)
                print(f"Answer: {answer}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()