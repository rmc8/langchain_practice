import argparse

from libs.data import Goal
from libs.model import get_model
from libs.proc import PassiveGoalCreator


def main():
    parser = argparse.ArgumentParser(
        description="PassiveGoalCreatorを利用して目標を生成します",
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()
    llm = get_model()
    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query=args.task)
    print(result.text)


if __name__ == "__main__":
    main()
