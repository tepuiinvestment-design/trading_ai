from chromadb.utils import embedding_functions

_embedding_function = None


def get_embedding_function():
    """Module-level singleton — the embedding model loads once per process,
    not once per request."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function
