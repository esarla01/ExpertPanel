"""Pydantic models for panel input and output."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Panel modes
# ---------------------------------------------------------------------------

class PanelMode(str, Enum):
    FIXED = "fixed"        # Three pre-defined personas from personas.yaml
    DYNAMIC = "dynamic"    # Personas generated to fit the question's tensions
    BASELINE = "baseline"  # Same persona sampled 3x (null-diversity control)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PanelRequest(BaseModel):
    question: str = Field(..., description="The question to put to the panel")
    mode: PanelMode = Field(
        default=PanelMode.DYNAMIC,
        description=(
            "Which persona strategy to use. "
            "'dynamic' generates personas fitted to the question (default). "
            "'fixed' uses the three pre-defined personas. "
            "'baseline' samples one persona three times as a diversity control."
        ),
    )


# ---------------------------------------------------------------------------
# Response components
# ---------------------------------------------------------------------------

class PersonaPosition(BaseModel):
    name: str
    role: str
    country: str
    position: str = Field(..., description="Their final position in 1-2 sentences")
    reasoning: str = Field(..., description="Why they hold this position in 1-2 sentences")
    confidence: str = Field(
        default="unknown",
        description="Self-reported confidence: high, moderate, or low",
    )
    changed_position: bool = Field(
        ..., description="Whether they shifted after hearing other views"
    )
    change_reason: Optional[str] = Field(
        default=None,
        description="If they changed position, the specific argument that convinced them",
    )


class Disagreement(BaseModel):
    topic: str
    sides: dict[str, str] = Field(
        ..., description="Mapping of persona name to their stance"
    )
    crux: str = Field(
        ..., description="The core reason for the disagreement"
    )


class UnresolvedQuestion(BaseModel):
    question: str = Field(..., description="A question the panel could not settle")
    why: str = Field(..., description="Why it remains open")


class PanelResult(BaseModel):
    question: str
    mode: str = Field(..., description="Which persona strategy was used")
    personas: list[PersonaPosition]
    disagreements: list[Disagreement]
    unresolved: list[UnresolvedQuestion] = Field(
        default_factory=list,
        description="Questions the panel could not settle",
    )
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
