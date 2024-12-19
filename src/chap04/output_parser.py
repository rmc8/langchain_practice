from typing import List

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Recipe(BaseModel):
    ingredients: List[str] = Field(description="ingredients of the dish")
    steps: List[str] = Field(description="Steps to make the dish")


def main():
    output_parser = PydanticOutputParser(pydantic_object=Recipe)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "ユーザーが入力した料理のレシピを考えてください。\n\n"
                "{format_instructions}",
            ),
            ("human", "{dish}"),
        ]
    )
    prompt_with_format_instructions = prompt.partial(
        format_instructions=format_instructions,
    )
    prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})

    model = ChatOllama(
        model="qwen2.5:14b",
        base_url="http://192.168.11.34:11434",
    )
    ai_message = model.invoke(prompt_value)
    # print(ai_message.content)
    recipe = output_parser.invoke(ai_message)
    print(recipe)


if __name__ == "__main__":
    main()
