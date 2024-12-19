from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    ingredients: List[str] = Field(description="ingredients of the dish")
    instructions: str = Field(description="Steps to make the dish")


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
    output_parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "ユーザーが入力した料理のレシピを考えてください。"
                "\n\n{format_instructions}",
            ),
            ("human", "{dish}"),
        ]
    )
    prompt_with_format_instructions = prompt.partial(
        format_instructions=output_parser.get_format_instructions()
    )
    model = get_model()
    chain = prompt_with_format_instructions | model | output_parser
    recipe = chain.invoke({"dish": "カレー"})
    print(recipe)


if __name__ == "__main__":
    main()
