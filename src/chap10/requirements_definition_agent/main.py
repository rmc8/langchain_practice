import argparse

# from langchain_openai import ChatOpenAI

from libs.model import get_model
from libs.agent import DocumentationAgent


def main():
    parser = argparse.ArgumentParser(
        description="ユーザー要求に基づいて要件定義を生成します。"
    )
    # Add task argument
    parser.add_argument(
        "--task", type=str, help="作成したいアプリケーションについて記載してください。"
    )
    # Add k argument
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成するペルソナの人数を設定してください。（デフォルト：5)",
    )
    # Parse arguments
    args = parser.parse_args()
    # Get model
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    llm = get_model()
    # Create DocumentationAgent instance
    agent = DocumentationAgent(llm=llm, k=args.k)
    # Run the agent with user request
    final_output = agent.run(user_request=args.task)
    print(final_output)


if __name__ == "__main__":
    main()
