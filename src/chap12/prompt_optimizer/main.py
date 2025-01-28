import argparse

from libs.model import get_model
from libs.proc import OptimizedGoal, PromptOptimizer


def main():
    parser = argparse.ArgumentParser(
        description="PromptOptimizerを利用して、生成された目標のリストを最適化します。"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()
    llm = get_model()
    passive_goal_creator = PromptOptimizer(llm)
    goal: OptimizedGoal = passive_goal_creator.run(query=args.task)
    print(goal.text)


if __name__ == "__main__":
    main()
