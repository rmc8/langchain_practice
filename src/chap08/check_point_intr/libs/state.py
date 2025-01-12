import operator
from typing import Annotated, Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field

from .model import get_model


class State(BaseModel):
    query: str
    messages: Annotated[List[BaseMessage], operator.add] = Field(default=[])


def add_message(state: State) -> Dict[str, Any]:
    additional_messages = []
    if not state.messages:
        additional_messages.append(
            SystemMessage(content="あなたは最小限の応答をする対話エージェントです。")
        )
    additional_messages.append(HumanMessage(content=state.query))
    return {"messages": additional_messages}


def llm_response(state: State) -> Dict[str, Any]:
    llm = get_model()
    ai_message = llm.invoke(state.messages)
    return {"messages": [ai_message]}
