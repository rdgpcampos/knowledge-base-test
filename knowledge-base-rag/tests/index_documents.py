#!/usr/bin/env python3

import os

from src.rag.vector_db_providers.init_vector_db import init_vector_db

def main():
    # Initialize Vector DB Provider
    vector_db_provider = init_vector_db()
    
    # Index files from a specific directory
    print("Indexing documents...")
    try:
        for dirpath, dirnames, _ in os.walk("/Users/rodrigocampos/knowledge-base-rag/documents/"):
            for dirname in dirnames:
                vector_db_provider.index_files(dirname,os.path.join(dirpath, dirname,"*.txt"))  # Adjust path as needed
    except Exception as e:
        print(f"Failed to index documents: {e}")
        raise e
    
    print("Files indexed")

if __name__ == "__main__":
    main()