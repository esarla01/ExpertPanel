"""Pydantic models for panel input and output."""

from pydantic import BaseModel, Field


class PanelRequest(BaseModel):
    question: str = Field(..., description="The question to put to the panel")


class PersonaPosition(BaseModel):
    name: str
    role: str
    country: str
    position: str = Field(..., description="Their final position on the question")
    reasoning: str = Field(..., description="Why they hold this position")
    changed_position: bool = Field(
        ..., description="Whether they shifted after hearing other views"
    )


class Disagreement(BaseModel):
    topic: str
    sides: dict[str, str] = Field(
        ..., description="Mapping of persona name to their stance"
    )
    crux: str = Field(
        ..., description="The core reason for the disagreement"
    )


class PanelResult(BaseModel):
    question: str
    personas: list[PersonaPosition]
    disagreements: list[Disagreement]
    raw_round_1: dict[str, str] = Field(
        default_factory=dict,
        description="Each persona's unedited first-round answer",
    )
    raw_round_2: dict[str, str] = Field(
        default_factory=dict,
        description="Each persona's unedited rebuttal",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Any failures that occurred during the run",
    )
