from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


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


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーが入力した料理のレシピを考えてください。"),
            ("human", "{dish}"),
        ]
    )
    model = get_model()
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    outputs = chain.batch([{"dish": "寿司"}, {"dish": "餃子"}])
    print(outputs)


if __name__ == "__main__":
    main()
