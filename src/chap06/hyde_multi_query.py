import os
from typing import List

from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


class QueryGenerationOutput(BaseModel):
    queries: List[str] = Field(..., description="検索クエリのリスト")


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
    retriever = db.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        以下の文脈だけを踏まえて質問に回答してください。
        文脈: '''
        {context}
        '''
        質問: {question}
        """.strip()
    )

    query_generation_prompt = ChatPromptTemplate.from_template(
        """\
        質問に対してベクターデータベースから関連文書を検索するために、
        ３つの異なる検索クエリを生成してください。
        距離ベースの類似性検索の限界を克服するために、
        ユーザーの質問に対して複数の視点を提供することが目標です。
        質問： {question}
        """
    )
    query_generation_chain = (
        query_generation_prompt
        | model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )
    multi_query_rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": query_generation_chain | retriever.map(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    res = multi_query_rag_chain.invoke("Langchainの概要を教えてください。")
    print(res)


if __name__ == "__main__":
    main()
