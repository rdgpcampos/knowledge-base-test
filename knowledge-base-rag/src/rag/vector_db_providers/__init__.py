from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBProvider(ABC):
    """Vector database interface"""
    @abstractmethod
    def index_files(self, tag: str, file_pattern: str):
        """Index files into vector database"""

    @abstractmethod
    def search_similar(self, tag: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for nodes that are similar to the query within the vector database"""