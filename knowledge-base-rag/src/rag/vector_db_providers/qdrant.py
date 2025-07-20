"""
Defining Qdrant as a VectorDBProvider implementation
"""

import os
import glob
import uuid
from typing import List, Dict, Any
from pathlib import Path
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
from . import VectorDBProvider


# Load environment variables
load_dotenv()

class Qdrant(VectorDBProvider):
    """Defines Qdrant as a Vector DB provider"""
    def __init__(self, collection_name: str, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize the RAG system with Qdrant and OpenAI clients."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-3.5-turbo"
        self.tokenizer = tiktoken.encoding_for_model(self.chat_model)

    def _chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
       
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def index_files(self, tag: str, file_pattern: str):
        """Index all text files matching the pattern."""
        files = glob.glob(file_pattern)
        
        if not files:
            print(f"No files found matching pattern: {file_pattern}")
            return
        
        print(f"Found {len(files)} files to index")
        
        points = []
        for file_path in files:
            print(f"Processing: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = self._chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    # Get embedding
                    embedding = self._get_embedding(chunk)
                    
                    # Create point
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "tag": tag,
                            "file_path": file_path,
                            "file_name": Path(file_path).name,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    points.append(point)
                
                print(f"  Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Upload to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Successfully indexed {len(points)} chunks")
    
    def search_similar(self, tag: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar text chunks."""
        query_embedding = self._get_embedding(query)
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tag",
                    match=models.MatchValue(value=tag)
                )
            ]
        )
        )
        
        results = []
        for result in search_results:
            results.append({
                "text": result.payload["text"],
                "file_name": result.payload["file_name"],
                "file_path": result.payload["file_path"],
                "chunk_index": result.payload["chunk_index"],
                "score": result.score
            })
        
        return results