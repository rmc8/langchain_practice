from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    chain = prompt | model | StrOutputParser()
    output = chain.invoke({"dish": "カレー"})
    print(output)


if __name__ == "__main__":
    main()
