import os
import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


def get_model(
    model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b"),
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature: float = 0.0,
    **kwargs,
) -> ChatOllama:
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )
    return llm


class Goal(BaseModel):
    description: str = Field(..., description="Goal description")

    @property
    def text(self) -> str:
        return str(self.description)


class PassiveGoalCreator:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件：\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "  - インターネットを利用して、目標を達成するための調査を行う。\n"
            "  - ユーザーのためのレポートを生成する。\n"
            "3. 決して2.意外の行動を取ってはいけません。\n"
            "ユーザーの入力： {query}"
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})


class OptimizedGoal(BaseModel):
    description: str = Field(..., description="目標の設定")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.description}(測定基準: {self.metrics})"


class PromptOptimizer:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            "あなたは目標設定の専門家です。以下の目標をSMART原則(Specific, Measurable, Achievable, Relevant, Time-bound)に基づいて最適化してください。\n\n"
            "元の目標\n"
            "{query}\n\n"
            "指示:\n"
            "1. 元の目標を分析し、不足している要素や改善点を特定してください。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "  - インターネットを利用して、目標を達成するための調査を行う\n"
            "  - ユーザーのためのレポートを生成する。\n"
            "3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。\n"
            "  -  一切抽象的な表現を含んではいけません。\n"
            "  - 必ず∀単語が実行可能かつ具体的であることを確認してください。\n"
            "4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。\n"
            "5. 元の目標で機嫌が指定されていない場合は、機嫌を考慮する必要はありません。\n"
            "6. REMEMBER: 決して2.以外の行動を取ってはいけません。"
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query})


class ResponseOptimizer:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。与えられた目標に対して、エージェントが目標にあったレスポンスを返すためのレスポンス仕様を定義してください。",
                ),
                (
                    "human",
                    "以下の手順に従って、レスポンス最適化プロンプトを作成してください:\n\n"
                    "1. 目標分析:\n"
                    "提示された目標を分析し、主要な要素や意図を特定してください。\n\n"
                    "2. レスポンス仕様の策定:\n"
                    "目標達成のための最適なレスポンス仕様を考案してください。トーン、構造、内容の焦点などを考慮にいれてください。\n\n"
                    "3. 具体的な指示の作成\n"
                    "事前に収集された上右方から、ユーザーの機体に沿ったレスポンスをするために必要なAIエージェントに対する明確で実行可能な支持を作成してください。あなたの指示によってAIエージェントが実行可能なのは、既に調査済みの結果をまとめることだけです。インターネットへのアクセスはできません。\n\n"
                    "4. 例の提供:\n"
                    "可能であれば、目標に沿ったレスポンスの例を1つ以上ふくめてください。\n\n"
                    "5. 評価基準の設定：\n"
                    "レスポンスの効果を測定するための基準を定義してください。\n\n"
                    "以下の構造でレスポンス最適化プロンプトを出力してください。\n\n"
                    "目標分析：\n"
                    "[ここに目標の分析結果を記入]\n\n"
                    "レスポンス仕様：\n"
                    "[ここにAIエージェントへの具体的な支持を記入]\n\n"
                    "レスポンス例：\n"
                    "[ここにレスポンス例を記入]\n\n"
                    "評価基準：\n"
                    "[ここに評価基準を記入]\n\n"
                    "では、以下の目標に対するレスポンス最適化プロンプトを作成してください。\n"
                    "{query}",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})


def main():
    parser = argparse.ArgumentParser(
        description="ResponseOptimizerを利用して、与えられた目標に対して最適化されたレスポンスの定義を生成します。"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()
    llm = get_model()
    passive_goal_creator = PassiveGoalCreator(llm)
    goal: Goal = passive_goal_creator.run(query=args.task)
    print(goal.text)
    prompt_optimizer = PromptOptimizer(llm)
    optimized_prompt = prompt_optimizer.run(goal.text)
    print(optimized_prompt)
    response_optimizer = ResponseOptimizer(llm)
    response = response_optimizer.run(query=optimized_prompt)
    print(response)


if __name__ == "__main__":
    main()
