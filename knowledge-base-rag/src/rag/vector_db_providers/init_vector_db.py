"""
Vector database initializer
If we change the vector db provider, modifying the settings here will
reflect the new vector db provider throughout the code
"""

from .qdrant import Qdrant
from . import VectorDBProvider

def init_vector_db() -> VectorDBProvider:
    """
    Initialize the vector db provider
    If a different vector db provider needs to be set, do it here
    """
    vector_db_provider = Qdrant("text_documents", "localhost", 6333)
    return vector_db_provider