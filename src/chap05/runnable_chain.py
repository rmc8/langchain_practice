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
    model = get_model()
    output_parser = StrOutputParser()
    cot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーの質問にステップバイステップで回答してください。"),
            ("human", "{question}"),
        ]
    )
    cot_chain = cot_prompt | model | output_parser
    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "ステップバイステップで考えた回答から結論だけを抽出してください。",
            ),
            ("human", "{question}"),
        ]
    )
    summarize_chain = summarize_prompt | model | output_parser

    cot_summarize_chain = cot_chain | summarize_chain
    res = cot_summarize_chain.invoke({"question": "10 + 2 * 3"})
    print(res)


if __name__ == "__main__":
    main()
