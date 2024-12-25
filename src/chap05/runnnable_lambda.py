from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


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
    chain = prompt | model | output_parser | RunnableLambda(upper)
    output = chain.invoke({"input": "Hello!"})
    print(output)


if __name__ == "__main__":
    main()
