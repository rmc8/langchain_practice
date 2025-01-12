from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from libs.state import State, add_message, llm_response
from libs.checkpoint import print_checkpoint_dump

ADD_MESSAGE = "add_Message"
LLM_RESPONSE = "llm_response"


def main():
    # Initialize the state graph and set up nodes.
    graph = StateGraph(State)
    graph.add_node(ADD_MESSAGE, add_message)
    graph.add_node(LLM_RESPONSE, llm_response)
    # Set the entry point and add edges to connect nodes.
    graph.set_entry_point(ADD_MESSAGE)
    graph.add_edge(ADD_MESSAGE, LLM_RESPONSE)
    graph.add_edge(LLM_RESPONSE, END)
    # Initialize the memory saver for checkpointing.
    checkpointer = MemorySaver()
    # Compile
    compiled_graph = graph.compile(checkpointer=checkpointer)
    # Invoke the graph with a query and configuration.
    config = {"configurable": {"thread_id": "example-1"}}
    user_query = State(query="私の好きなものはずんだ餅です。覚えておいてね！")
    first_response = compiled_graph.invoke(user_query, config)
    # print(first_response)
    # print_checkpoint_dump(checkpointer, config)
    user_query = State(query="私の好物は何か覚えていますか？")
    second_response = compiled_graph.invoke(user_query, config)
    print(second_response)    

if __name__ == "__main__":
    main()
