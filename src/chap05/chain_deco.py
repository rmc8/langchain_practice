import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain


def get_model(
    model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b"),
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
):
    from langchain_ollama import ChatOllama

    print(base_url, "TEST")
    model = ChatOllama(
        model=model,
        base_url=base_url,
    )
    return model


@chain
def upper(text: str) -> str:
    return text.upper()


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )
    model = get_model()
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser | upper
    output = chain.invoke({"input": "Hello!"})
    print(output)


if __name__ == "__main__":
    main()
