from pprint import pprint

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

def print_checkpoint_dump(checkpointer: BaseCheckpointSaver, config: RunnableConfig):
    checkpoint_tuple = checkpointer.get_tuple(config)
    print("チェックポイントデータ：")
    pprint(checkpoint_tuple.checkpoint)
    print("\nメタデータ：")
    pprint(checkpoint_tuple.metadata)