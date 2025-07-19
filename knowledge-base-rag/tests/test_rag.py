#!/usr/bin/env python3
"""
Example usage of the RAG system
"""

from src.rag.rag import RAGSystem

def main():
    # Initialize the RAG system
    rag = RAGSystem(qdrant_host="localhost", qdrant_port=6333)
    
    # Index files from a specific directory
    print("Indexing text files...")
    rag.index_files("/Users/rodrigocampos/knowledge-base-rag/documents/*.txt")  # Adjust path as needed
    
    # Or index files from current directory
    # rag.index_files("*.txt")
    
    # Example queries
    queries = [
        "What is the main topic discussed in the documents?",
        "Can you summarize the key points?",
        "What are the important dates mentioned?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        answer = rag.query_with_rag(query)
        print(f"Answer: {answer}")
        print("-" * 50)
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode - Enter your questions:")
    
    while True:
        question = input("\nYour question (or 'quit' to exit): ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question.strip():
            answer = rag.query_with_rag(question)
            print(f"Answer: {answer}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()