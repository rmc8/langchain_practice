import operator
from typing import Annotated, List

from pydantic import BaseModel, Field

from .data import Interview, Persona


class InterviewState(BaseModel):
    user_request: str = Field(..., description="User's request for the interview")
    personas: Annotated[List[Persona], operator.add] = Field(default_factory=list, description="List of generated personas")
    interviews: Annotated[List[Interview], operator.add] = Field(default_factory=list, description="List of conducted interviews")
    requirements_doc:str = Field(default="", description="Generated requirements document")
    iteration :int = Field(default=0, description="Count of iterations for the interview process")
    is_information_sufficient:bool = Field(default=False, description="Ιs the information sufficient?")
