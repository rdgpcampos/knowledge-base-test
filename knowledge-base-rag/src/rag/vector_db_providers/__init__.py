from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBProvider(ABC):
    @abstractmethod
    def index_files(self, tag: str, file_pattern: str):
        pass

    @abstractmethod
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        pass