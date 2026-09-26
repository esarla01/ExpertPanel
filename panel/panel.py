"""Panel orchestration: runs the three-persona debate and extracts structure."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from panel.llm import generate
from panel.schemas import PanelResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load personas
# ---------------------------------------------------------------------------

_PERSONAS_PATH = Path(__file__).resolve().parent.parent / "personas.yaml"


def _load_personas() -> list[dict[str, str]]:
    with open(_PERSONAS_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _system_prompt(persona: dict) -> str:
    return (
        f"You are {persona['name']}, a {persona['role']} based in "
        f"{persona['country']}.\n\n"
        f"{persona['perspective']}\n\n"
        "Stay in character throughout. Be specific and direct. "
        "Do not hedge excessively or try to please everyone."
    )


def _round1_prompt(question: str) -> str:
    return (
        f"A panel of senior professionals is discussing this question:\n\n"
        f"{question}\n\n"
        "Give your independent answer. State your position clearly, "
        "explain your reasoning, and identify the trade-offs you accept."
    )


def _round2_prompt(
    question: str,
    persona_name: str,
    other_answers: dict[str, str],
) -> str:
    others_text = "\n\n---\n\n".join(
        f"**{name}** said:\n{answer}"
        for name, answer in other_answers.items()
    )
    return (
        f"The question was:\n\n{question}\n\n"
        f"Your colleagues responded as follows:\n\n{others_text}\n\n"
        f"You are {persona_name}. Respond to your colleagues. "
        "Where do you agree? Where do you disagree, and why? "
        "If you have changed your position, say so and explain what "
        "convinced you. If you have not changed, say why not."
    )


_SUMMARY_SYSTEM = (
    "You are an impartial moderator summarising a panel discussion. "
    "Return ONLY valid JSON matching the schema below. No markdown, "
    "no commentary, no code fences.\n\n"
    "Schema:\n"
    "{\n"
    '  "personas": [\n'
    "    {\n"
    '      "name": "...",\n'
    '      "role": "...",\n'
    '      "country": "...",\n'
    '      "position": "their final position in 1-2 sentences",\n'
    '      "reasoning": "why they hold it in 1-2 sentences",\n'
    '      "changed_position": true/false\n'
    "    }\n"
    "  ],\n"
    '  "disagreements": [\n'
    "    {\n"
    '      "topic": "short label",\n'
    '      "sides": {"Name1": "stance", "Name2": "stance"},\n'
    '      "crux": "core reason for the disagreement"\n'
    "    }\n"
    "  ]\n"
    "}"
)


def _summary_prompt(
    question: str,
    round1: dict[str, str],
    round2: dict[str, str],
) -> str:
    parts = []
    for name in round1:
        parts.append(
            f"## {name}\n\n"
            f"### Opening position\n{round1[name]}\n\n"
            f"### After discussion\n{round2.get(name, '[no response]')}"
        )
    transcript = "\n\n---\n\n".join(parts)
    return (
        f"The panel discussed this question:\n\n{question}\n\n"
        f"Here is the full transcript:\n\n{transcript}\n\n"
        "Produce the JSON summary now."
    )


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_summary(raw: str) -> dict[str, Any]:
    """Try to parse the summary JSON, stripping common model quirks."""
    text = raw.strip()
    # Strip markdown code fences if the model adds them despite instructions
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def _extract_summary(
    question: str,
    round1: dict[str, str],
    round2: dict[str, str],
    errors: list[str],
) -> dict[str, Any]:
    """Ask the model to summarise the transcript as JSON, with one repair retry."""
    raw = generate(
        _summary_prompt(question, round1, round2),
        system=_SUMMARY_SYSTEM,
        use_cache=False,  # always fresh summary
    )
    try:
        return _parse_summary(raw)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Summary parse failed (%s), retrying...", exc)

    repair_prompt = (
        f"Your previous response was not valid JSON:\n\n{raw}\n\n"
        "Fix it. Return ONLY valid JSON matching the schema."
    )
    try:
        raw = generate(repair_prompt, system=_SUMMARY_SYSTEM, use_cache=False)
        return _parse_summary(raw)
    except Exception as exc:
        msg = f"Summary extraction failed after retry: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {"personas": [], "disagreements": []}


# ---------------------------------------------------------------------------
# Run the panel
# ---------------------------------------------------------------------------

def run_panel(question: str) -> PanelResult:
    """Execute a full panel discussion and return structured results."""
    personas = _load_personas()
    errors: list[str] = []

    # ---- Round 1: independent answers (parallel) -------------------------
    def _round1_answer(persona: dict) -> tuple[str, str]:
        try:
            answer = generate(_round1_prompt(question), system=_system_prompt(persona))
        except Exception as exc:
            msg = f"Round 1 failed for {persona['name']}: {exc}"
            logger.error(msg)
            errors.append(msg)
            answer = "[This persona was unavailable.]"
        return persona["name"], answer

    with ThreadPoolExecutor(max_workers=3) as pool:
        round1 = dict(pool.map(_round1_answer, personas))

    # ---- Round 2: rebuttals (parallel) -----------------------------------
    def _round2_answer(persona: dict) -> tuple[str, str | None]:
        other_answers = {
            name: text for name, text in round1.items() if name != persona["name"]
        }
        prompt = _round2_prompt(question, persona["name"], other_answers)
        try:
            answer = generate(prompt, system=_system_prompt(persona))
        except Exception as exc:
            msg = f"Round 2 failed for {persona['name']}: {exc}"
            logger.error(msg)
            errors.append(msg)
            return persona["name"], None
        return persona["name"], answer

    with ThreadPoolExecutor(max_workers=3) as pool:
        round2 = {
            name: answer
            for name, answer in pool.map(_round2_answer, personas)
            if answer is not None
        }

    # ---- Summary extraction ----------------------------------------------
    summary = _extract_summary(question, round1, round2, errors)

    return PanelResult(
        question=question,
        personas=summary.get("personas", []),
        disagreements=summary.get("disagreements", []),
        raw_round_1=round1,
        raw_round_2=round2,
        errors=errors,
    )
