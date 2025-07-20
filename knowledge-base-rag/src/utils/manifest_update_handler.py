from src.rag.rag import QueryController

def update_manifest(rag: QueryController, feedback: str):
    print(rag.generate_manifest_change(feedback))