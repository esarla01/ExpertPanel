# Synthetic Expert Panel

A small service that puts a question to three AI-simulated medical experts,
runs a structured debate, and returns each expert's position, reasoning
and the points where they disagree.

## Quick start

TODO: setup and deploy instructions

You run it every time you want to update what's live:
gcloud run deploy panel --source . --allow-unauthenticated

To test locally: 
uvicorn app:app --reload

## Design decisions


The flow: 

Client sends POST /panel {"question": "..."}
  │
  ▼
FastAPI parses into PanelRequest
  │
  ▼
run_panel(question)
  │
  ├── Load personas.yaml
  │
  ├── Round 1 (3 calls in parallel)
  │     Each: generate() → cache check → Gemini API → retry if needed
  │
  ├── Round 2 (3 calls in parallel)
  │     Each sees the other two round-1 answers
  │
  ├── Summary call (1 call)
  │     Moderator extracts structured JSON
  │
  ├── Parse JSON (retry once if malformed)
  │
  └── Return PanelResult
  │
  ▼
FastAPI serializes to JSON, sends HTTP response

Returns:

PanelResult(
    question="Who should get GLP-1 treatment first?",
    personas=[...],         # structured positions from the summary
    disagreements=[...],    # structured disagreements from the summary
    raw_round_1={           # the unedited text from each persona
        "Dr Helen Carter": "...",
        "Dr Marcus Reed": "...",
        "Dr Ana Souza": "...",
    },
    raw_round_2={...},      # the unedited rebuttals
    errors=[],              # empty if everything worked
)


## AI tools used

TODO


