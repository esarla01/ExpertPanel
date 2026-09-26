"""FastAPI service exposing the synthetic expert panel."""

import logging

from fastapi import FastAPI

from panel.panel import run_panel
from panel.schemas import PanelRequest, PanelResult

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Synthetic Expert Panel",
    description="Put a question to a panel of three AI-simulated experts.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/panel", response_model=PanelResult)
def panel(req: PanelRequest):
    return run_panel(req.question)
