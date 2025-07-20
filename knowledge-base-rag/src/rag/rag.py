"""
Query controller that processes user messages
The QueryController class is agnostic to the vector db provider,
so changing the vector db provider should not affect the behavior of this class
as long as the vector db provider correctly implements the VectorDBProvider interface
"""

import os
from enum import StrEnum, auto
import json
from typing import Dict
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from src.rag.vector_db_providers import VectorDBProvider

# Load environment variables
load_dotenv()

class MessageType(StrEnum):
    """Types of messages received from the user"""
    QUERY = auto()
    FEEDBACK = auto()
    OTHER = auto()

class QueryController:
    """Defines methods to process the user query using AI"""
    def __init__(self, vector_db_provider: VectorDBProvider):
        """Initialize the RAG system with Qdrant and OpenAI clients."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_model = "gpt-3.5-turbo"
        self.vector_db_provider = vector_db_provider
        self.tokenizer = tiktoken.encoding_for_model(self.chat_model)

    def feedback_or_query(self, message: str) -> Dict[str, str]:
        """Determine if a user message is a query or a feedback"""
        prompt = f"""
Analyze if the following message is a feedback or a query.
Use the definitions below to know how to differentiate between the two:
- Query: 
    A question where the user wants to know some type of information.
- Feedback: 
    A comment where the user instructs you on how to improve the quality of your responses. 
    Sometimes this can come in the form of a question, where the user is politely asking if you can improve your responses in a given way.

Examples:
You should make your responses longer - feedback
What is the capital of France? - query
What is the difference between a neutron and a proton? - query
Can you include citations in your responses? - feedback

If you don't know whether the message is a feedback or a query, classify it as 'Other'.

# MESSAGE #
{message}
##########

If the message is a feedback, you should edit it to look like a prompt targeted towards LLMs, and your response should follow the JSON structure below:
When editting the message to make it look like a prompt, make sure that the message is specific, direct, and concise.

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
        """Modify manifest"""
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
If the feedback enters in contradiction with some part of the document, favor the feedback over the contradicting part.
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

    
    def query_with_rag(self, tag: str, question: str, max_context_length: int = 3000) -> str:
        """Query using RAG: retrieve relevant chunks and generate answer."""
        # Search for relevant chunks
        search_results = self.vector_db_provider.search_similar(tag, question, limit=10)
        
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
            
            reference_path = os.path.join(dirname, "../../../documents",tag,"template.md")
            with open(reference_path, "r", encoding="utf-8") as reference_file:
                reference = reference_file.read()
        except Exception as e:
            print(f"Failed to read manifest: {e}")
            raise e
        
        prompt = manifest.format(information=context,query=question, reference=reference)
        
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