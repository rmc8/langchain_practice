from langgraph.graph import StateGraph
from langgraph.graph import END

from libs.state import State
from libs.node import selection_node, answering_node, check_node

SELECTION_NODE = "selection"
ANSWERING_NODE = "answering"
CHECK_NODE = "check"


def main():
    # ワークフローの定義
    workflow = StateGraph(State)
    workflow.add_node(SELECTION_NODE, selection_node)
    workflow.add_node(ANSWERING_NODE, answering_node)
    workflow.add_node(CHECK_NODE, check_node)
    # エッジの定義
    workflow.set_entry_point(SELECTION_NODE)
    workflow.add_edge(SELECTION_NODE, ANSWERING_NODE)
    workflow.add_edge(ANSWERING_NODE, CHECK_NODE)
    # 条件付エッジの定義
    workflow.add_conditional_edges(
        CHECK_NODE,
        lambda state: state.current_judge,
        {True: END, False: SELECTION_NODE},
    )
    # 実行
    compiled = workflow.compile()
    initial_state = State(query="生成AIについて教えてください。")
    result = compiled.invoke(initial_state)
    print(result)


if __name__ == "__main__":
    main()
