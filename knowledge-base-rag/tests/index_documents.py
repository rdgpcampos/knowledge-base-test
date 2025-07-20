"""
Run this script to index the documents in the vector db
We assume that the documents dir has the following structure:

knowledge-base-test/documents
    - document-category-1
        - a.txt
        - b.txt
        ...
    - document-category-2
        - c.txt
        - d.txt
        - e.txt
        ...

During indexing, the document category is included as a tag in each vector
This is later used to filter the documents to fetch while 
looking for vectors that are similar to the user query
"""

#!/usr/bin/env python3

import os

from src.rag.vector_db_providers.init_vector_db import init_vector_db

def main():
    """Index documents"""
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