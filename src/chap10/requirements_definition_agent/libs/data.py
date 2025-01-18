from typing import List

from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str = Field(..., description="The name of the persona.")
    background: str = Field(..., description="This persona's background or history.")


class Personas(BaseModel):
    personas: List[Persona] = Field(
        default_factory=list, description="A list of Persona objects."
    )


class Interview(BaseModel):
    persona: Persona = Field(..., description="The persona being interviewed.")
    question: str = Field(..., description="The interview question.")
    answer: str = Field(..., description="The interviewee's response.")


class InterviewResult(BaseModel):
    interviews: List[Interview] = Field(
        default_factory=list, description="A list of Interview objects."
    )


class EvaluationResult(BaseModel):
    reason: str = Field(..., description="The reason for the evaluation.")
    is_sufficient: bool = Field(
        ..., description="Whether the evaluation was sufficient or not."
    )
