from pydantic import BaseModel, Field


class Goal(BaseModel):
    description: str = Field(..., description="Goal description")

    @property
    def text(self) -> str:
        return str(self.description)
