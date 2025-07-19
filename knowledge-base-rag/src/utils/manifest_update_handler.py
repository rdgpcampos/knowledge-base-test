from src.rag.rag import RAGSystem

def update_manifest(rag: RAGSystem, feedback: str):
    print(rag.generate_manifest_change(feedback))