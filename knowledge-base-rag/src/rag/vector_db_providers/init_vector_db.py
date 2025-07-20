from .qdrant import Qdrant
from . import VectorDBProvider

def init_vector_db() -> VectorDBProvider:
    vector_db_provider = Qdrant("text_documents", "localhost", 6333)
    return vector_db_provider