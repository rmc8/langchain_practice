from langchain_core.messages import AIMessage
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
    output_parser = StrOutputParser()
    ai_message = AIMessage(content="こんにちは。私はAIアシスタントです。")
    output = output_parser.invoke(ai_message)
    print(output)


if __name__ == "__main__":
    main()
