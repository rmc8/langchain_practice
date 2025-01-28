import os
import argparse
import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
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
            "あなたは目標設定の専門家です。以下の目標をSMART原則(Specific, Mesurable, Achievable, Relevant, Time-bound)に基づいて最適化してください。\n\n"
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


class TaskOption(BaseModel):
    description: str = Field(default="", description="タスクオプションの説明")


class Task(BaseModel):
    task_name: str = Field(..., description="タスクの名前")
    options: List[TaskOption] = Field(
        default_factory=list,
        min_items=2,
        max_items=3,
        description="2~3個のタスクオプション",
    )


class DecomposedTasks(BaseModel):
    values: List[Task] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="3~5個に分解されたタスク",
    )


class MultiPathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス")
    tasks: DecomposedTasks = Field(
        default_factory=DecomposedTasks,
        description="複数のオプションをもつタスクのリスト",
    )
    current_task_index: int = Field(default=0, description="現在のタスクのインデックス")
    chosen_options: Annotated[List[int], operator.add] = Field(
        default_factory=list,
        description="各タスクで選択されたオプションのインデックス",
    )
    results: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="実行されたタスクの結果",
    )
    final_output: str = Field(
        default="",
        description="最終的な出力",
    )


class QueryDecomposer:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "---\n"
            "タスク：与えられた目標を3~5個の高レベルタスクにブナつし、各タスクに2〜3個の具体的なオプションを提供してください。\n"
            "要件：\n"
            "1. 以下の行動だけで目標を達成すること。決して指定されていない行動をとらないこと。\n"
            "  - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各高レベルタスクは具体的かつ詳細に記載されており、単独で実行並びに検証可能な情報を含めること。一切抽象的な表現を含まないこと\n"
            "3. 各項レベルタスクに2~3個の異るアプローチ又はオプションを提供すること。\n"
            "4. タスクは実行可能な順序でリスト化すること。\n"
            "5. タスクは日本語で出力すること。\n\n"
            "REMEMBER: 実行できないタスク、ならびに選択肢は絶対に作成しないでください。\n\n"
            "目標： {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})


class OptionPresenter:
    def __init__(self, llm: ChatOllama):
        self.llm = llm.configurable_fields(
            max_tokens=ConfigurableField(id="max_tokens")
        )

    def run(self, task: Task) -> int:
        task_name = task.task_name
        options = task.options
        print(f"\nタスク： {task_name}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option.description}")
        choice_prompt = ChatPromptTemplate.from_template(
            "タスク：与えられたタスクとオプションに基づいて、最適なオプションを選択してください。必ず番号のみで回答してください。\n\n"
            "なお、あなたは以下の行動しかできません。\n"
            "  - インターネットを利用して目標を達成するための調査を行う。\n\n"
            "タスク： {task_name}\n"
            "オプション：\n{options_text}\n"
            "選択： (1-{num_options})"
        )
        option_text = "\n".join(
            f"{i}. {option.description}" for i, option in enumerate(options, 1)
        )
        chain = (
            choice_prompt
            | self.llm.with_config(config=dict(max_tokens=1))
            | StrOutputParser()
        )
        choice_str = chain.invoke(
            {
                "task_name": task_name,
                "options_text": option_text,
                "num_options": len(options),
            }
        )
        print(f"==> エージェントの選択： {choice_str}\n")
        return int(choice_str.strip()) - 1


class TaskExecutor:
    def _init__(self, llm: ChatOllama):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: Task, chosen_option: TaskOption) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        "以下のタスクを実行し、詳細な回答を提供してください：\n\n"
                        f"タスク：{task.task_name}\n"
                        f"選択されたアプローチ： {chosen_option.description}\n\n"
                        "要件：\n"
                        "1. 必要に応じて提供されたツールを使用すること。\n"
                        "2. 実行において徹底的かつ包括的であること。\n"
                        "3. 可能な限り具体的な事実やデータを提供すること。\n"
                        "4. 発見事項を明確にまとめること。\n",
                    )
                ]
            }
        )
        return result["messages"][-1].content


class ResultAggregator:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(
        self,
        query: str,
        response_definition: str,
        tasks: List[Task],
        chosen_options: List[int],
        results: List[str],
    ) -> str:
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標：\n{query}\n\n"
            "調査結果：\n{task_results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "{response_definition}"
        )
        task_results = self._format_task_results(tasks, chosen_options)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "task_results": task_results,
                "response_definition": response_definition,
            }
        )

    @staticmethod
    def _format_task_results(
        tasks: List[Task], chosen_options: List[int], results: List[str]
    ) -> str:
        task_results = ""
        for i, (task, chosen_option, result) in enumerate(
            zip(tasks, chosen_options, results)
        ):
            task_name = task.task_name
            chosen_option_desc = task.options[chosen_option].description
            task_results += f"タスク {i+1}: {task_name}\n"
            task_results += f"選択されたオプション: {chosen_option_desc}\n"
            task_results += f"結果: {result}\n\n"
        return task_results


class MultiPathPlanGeneration:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.response_optimizer = ResponseOptimizer(llm=self.llm)
        self.query_decomposer = QueryDecomposer(llm=self.llm)
        self.option_presenter = OptionPresenter(llm=self.llm)
        self.task_executor = TaskExecutor(llm=self.llm)
        self.result_aggregator = ResultAggregator(llm=self.llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        GOAL_SETTINGS = "goal_settings"
        DECOMPOSE_QUERY = "decompose_query"
        PRESENT_OPTIONS = "present_options"
        EXECUTE_TASK = "execute_task"
        AGGREGATE_REQUESTS = "aggregate_requests"
        graph = StateGraph()
        graph.add_node(GOAL_SETTINGS, self._goal_setting)
        graph.add_node(DECOMPOSE_QUERY, self._decompose_query)
        graph.add_node(PRESENT_OPTIONS, self._present_options)
        graph.add_node(EXECUTE_TASK, self._execute_task)
        graph.set_entry_point(GOAL_SETTINGS)
        graph.add_edge(GOAL_SETTINGS, DECOMPOSE_QUERY)
        graph.add_edge(DECOMPOSE_QUERY, PRESENT_OPTIONS)
        graph.add_edge(PRESENT_OPTIONS, EXECUTE_TASK)
        graph.add_conditional_edges(
            EXECUTE_TASK,
            lambda state: state.current_task_index < len(state.tasks.values),
            {True: PRESENT_OPTIONS, False: AGGREGATE_REQUESTS},
        )
        graph.add_edge(AGGREGATE_REQUESTS, END)
        return graph.compile()

    def _goal_setting(self, state: MultiPathPlanGenerationState) -> Dict[str, Any]:
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        return {
            "optimized_goal": optimized_goal,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: MultiPathPlanGenerationState) -> Dict[str, Any]:
        tasks = self.query_decomposer.run(query=state.optimized_goal)
        return {"tasks": tasks}

    def _present_options(self, state: MultiPathPlanGenerationState) -> Dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = self.option_presenter.run(task=current_task)
        return {"chosen_options": [chosen_option]}

    def _execute_task(self, state: MultiPathPlanGenerationState) -> Dict[str, Any]:
        current_task = state.tasks.values[state.current_task_index]
        chosen_option = current_task.options[state.chosen_options[-1]]
        result = self.task_executor.run(
            task=current_task,
            chosen_option=chosen_option,
        )
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(self, state: MultiPathPlanGenerationState) -> Dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            tasks=state.tasks.values,
            chosen_options=state.chosen_options,
            results=state.results,
        )
        return {"final_output": final_output}

    def run(self, query: str):
        initial_state = MultiPathPlanGeneration(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state.get("final_output", "最終的な回答の生成に失敗しました。")


def main():
    parser = argparse.ArgumentParser(
        description="MultiPathPlanGenerationを利用してタスクを実行します。"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()
    # llm = get_model()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    agent = MultiPathPlanGeneration(llm=llm)
    result = agent.run(query=args.task)
    print("\n=== 最終出力 ===")
    print(result)


if __name__ == "__main__":
    main()
