import os
from typing import Any, Dict

from langchain_core.runnables import RunnablePassthrough
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def rerank(inp: Dict[str, Any], top_n: int = 3):
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=top_n,
    )
    return cohere_reranker.compress_documents(documents=documents, query=question)


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def get_embeddings(
    model: str = os.environ.get("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:latest"),
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
):
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )
    return embeddings


def get_model(
    model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b"),
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
):
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        model=model,
        base_url=base_url,
    )
    return model


def get_vec_db():
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        file_filter=file_filter,
        branch="master",
    )
    documents = loader.load()
    embeddings = get_embeddings()
    db = Chroma.from_documents(documents, embeddings)
    return db


def main():
    model = get_model()
    db = get_vec_db()
    prompt = ChatPromptTemplate.from_template(
        """
        以下の文脈だけを踏まえて質問に回答してください。
        文脈: '''
        {context}
        '''
        質問: {question}
        """.strip()
    )
    retriever = db.as_retriever()
    rerank_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "documents": retriever,
        }
        | RunnablePassthrough.assign(context=rerank)
        | prompt
        | model
        | StrOutputParser()
    )
    res = rerank_rag_chain.invoke("LangChainの概要を教えてください。")
    print(res)


if __name__ == "__main__":
    main()
