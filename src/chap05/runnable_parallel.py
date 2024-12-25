import os

from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


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


def main():
    model = get_model()
    output_parser = StrOutputParser()
    # 楽観
    optimistic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください。",
            ),
            ("human", "{topic}"),
        ]
    )
    optimistic_chain = optimistic_prompt | model | output_parser
    # 悲観
    pessimistic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。",
            ),
            ("human", "{topic}"),
        ]
    )
    pessimistic_chain = pessimistic_prompt | model | output_parser
    # 楽観・悲観の意見を作成
    parallel_chain = RunnableParallel(
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain,
        }
    )
    # 2つの意見を集約する
    synthesize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
            (
                "human",
                "楽観的意見: {optimistic_opinion}\n悲観的意見: {pessimistic_opinion}",
            ),
        ]
    )
    synthesize_chain = parallel_chain | synthesize_prompt | model | output_parser
    output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
    print(output)


if __name__ == "__main__":
    main()
