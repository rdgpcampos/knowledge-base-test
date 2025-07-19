#!/usr/bin/env python3
"""
Example usage of the RAG system
"""

from src.rag.rag import RAGSystem

def main():
    # Initialize the RAG system
    rag = RAGSystem(qdrant_host="localhost", qdrant_port=6333)
    
    # Index files from a specific directory
    print("Indexing documents...")

    try:
        rag.index_files("/Users/rodrigocampos/knowledge-base-rag/documents/*.txt")  # Adjust path as needed
    except Exception as e:
        print(f"Failed to index documents: {e}")
        raise e
    
    print("Files indexed")

if __name__ == "__main__":
    main()