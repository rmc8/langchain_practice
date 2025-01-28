import os


from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_model(
    model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b"),
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature: float = 0.0,
    **kwargs
) -> ChatOllama:
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )
    return llm


def get_embeddings(
    model: str = os.environ.get("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:latest"),
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
) -> OllamaEmbeddings:
    embeddings = OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )
    return embeddings
