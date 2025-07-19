import os
import glob
from enum import StrEnum, auto
import json
from typing import List, Dict, Any
from pathlib import Path
import tiktoken
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

class MessageType(StrEnum):
    QUERY = auto()
    FEEDBACK = auto()
    OTHER = auto()

class RAGSystem:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize the RAG system with Qdrant and OpenAI clients."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "text_documents"
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-3.5-turbo"
        self.tokenizer = tiktoken.encoding_for_model(self.chat_model)
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection for storing embeddings."""
        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # text-embedding-3-small dimension
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
    
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
    
    def index_files(self, file_pattern: str = "*.txt"):
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
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar text chunks."""
        query_embedding = self._get_embedding(query)
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
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
    


    def feedback_or_query(self, message: str) -> Dict[str, str]:
        """Determine if a user message is a query or a feedback"""
        prompt = f"""
Analyze if the following message is a feedback or a query.

# MESSAGE #
{message}
##########

If the message is a feedback, you should edit it to look like a prompt targeted towards LLMs, and your response should follow the JSON structure below:

{{
"type": "{MessageType.FEEDBACK}",
"response": "[message editted as a prompt]"
}}

If the message is a query, your response should follow the JSON structure below:
{{
"type": "{MessageType.QUERY}",
"response": "[message as-is]"
}}

If the message is neither a query nor a feedback, your response should follow the JSON structure below:

{{
"type": "{MessageType.OTHER}",
"response": "[message as-is]"
}}

"""
        
        response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "user","content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
        
        try:
            response_json = json.loads(response.choices[0].message.content)
            return response_json

        except Exception as e:
            print(f"Could not determine the type of message: {e}")
            return {"type": MessageType.OTHER, "response": message}
        
    def generate_manifest_change(self, feedback: str) -> str:
        try:
            dirname = os.path.dirname(__file__)
            manifest_path = os.path.join(dirname, "../../../manifest/manifest.txt")
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = manifest_file.read()
        except Exception as e:
            print(f"Failed to read manifest: {e}")
            raise e
        
        prompt = f"""
You are an AI assistant specialized in modifying text documents according to user feedback.

Your task is to modify a document according to user feedback, while maintaining the document's overall structure.

The document below is a prompt that will be sent to an LLM. 
------
{manifest}
------
Your response should be in text format, including only the background information that you were provided.


The feedback below should be used to modify the document.
------
{feedback}
------

Modify the document according to the user feedback.
Do not remove or add any text enclosed by curly braces.

Your response should be simply the modified document.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "user","content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

    
    def query_with_rag(self, question: str, max_context_length: int = 3000) -> str:
        """Query using RAG: retrieve relevant chunks and generate answer."""
        # Search for relevant chunks
        search_results = self.search_similar(question, limit=10)
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from search results
        context_pieces = []
        total_tokens = 0
        
        for result in search_results:
            text = result["text"]
            tokens = len(self.tokenizer.encode(text))
            
            if total_tokens + tokens > max_context_length:
                break
            
            context_pieces.append(f"From {result['file_name']}:\n{text}")
            total_tokens += tokens
        
        context = "\n\n".join(context_pieces)
        
        # Generate answer using OpenAI
        try:
            dirname = os.path.dirname(__file__)
            manifest_path = os.path.join(dirname, "../../../manifest/manifest.txt")
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = manifest_file.read()
        except Exception as e:
            print(f"Failed to read manifest: {e}")
            raise e
        
        prompt = manifest.format(information=context,query=question)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "user","content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def clear_index(self):
        """Clear all indexed documents."""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._create_collection()
            print("Index cleared successfully")
        except Exception as e:
            print(f"Error clearing index: {e}")


def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example usage
    print("RAG System initialized")
    
    # Index files (adjust pattern as needed)
    print("\nIndexing files...")
    rag.index_files("*.txt")  # Index all .txt files in current directory
    
    # Example queries
    while True:
        print("\n" + "="*50)
        question = input("Enter your question (or 'quit' to exit): ")
        
        if question.lower() == 'quit':
            break
        
        print("\nSearching for relevant information...")
        answer = rag.query_with_rag(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()