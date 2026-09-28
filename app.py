"""FastAPI service exposing the synthetic expert panel."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from panel.panel import run_panel
from panel.schemas import PanelRequest, PanelResult

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Synthetic Expert Panel",
    description="Put a question to a panel of three AI-simulated experts.",
)

_FRONTEND_PATH = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/panel", response_model=PanelResult)
def panel(req: PanelRequest):
    return run_panel(req.question, mode=req.mode)


@app.get("/", response_class=HTMLResponse)
def frontend():
    return _FRONTEND_PATH.read_text()
