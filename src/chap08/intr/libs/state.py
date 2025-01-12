import operator
from typing import Annotated, List

from pydantic import BaseModel, Field


class State(BaseModel):
    query: str = Field(
        ...,
        description="ユーザーからの質問",
    )
    current_role: str = Field(
        default="",
        description="選定された回答ロール",
    )
    messages: Annotated[List[str], operator.add] = Field(
        default=[],
        description="回答履歴",
    )
    current_judge: bool = Field(
        default=False,
        description="品質チェックの結果",
    )
    judgement_reason: str = Field(
        default="",
        description="品質チェックの判定理由",
    )


class Judgement(BaseModel):
    reason: str = Field(
        default="",
        description="判定理由",
    )
    judge: bool = Field(
        default=False,
        description="判定結果",
    )
