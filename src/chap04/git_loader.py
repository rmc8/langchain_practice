from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter


def get_embeddings(
    model: str = "mxbai-embed-large:latest",
    base_url: str = "http://192.168.11.34:11434",
):
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )
    return embeddings


def get_model(
    model: str = "qwen2.5:14b",
    base_url: str = "http://192.168.11.34:11434",
):
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        model=model,
        base_url=base_url,
    )
    return model


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def main():
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter,
    )
    raw_docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(raw_docs)

    embeddings = get_embeddings()
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    query = "AWSのS3からデータを読み込むためのDocument loaderはありますか？"
    context_docs = retriever.invoke(query)
    first_doc = context_docs[0]
    # print(f"metadata = {first_doc.metadata}")
    # print(first_doc.page_content)

    prompt = ChatPromptTemplate.from_template(
        """
        以下の文脈だけ踏まえて質問に回答してください。
        
        文脈:'''
        {context}
        '''
        
        質問： {question}
        """
    )
    model = get_model()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    output = chain.invoke(query)
    print(output)


if __name__ == "__main__":
    main()
