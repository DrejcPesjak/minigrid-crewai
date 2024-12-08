from pydantic import BaseModel, Field

class AgentAction(BaseModel):
    explanation: str = Field(..., description="The explanation why should the action be taken")
    action: int = Field(..., description="The action to take")