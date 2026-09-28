# Synthetic Expert Panel

A service that puts a question to three AI-simulated experts, runs a
structured debate, and returns each expert's position, reasoning, and
the points where they disagree.

Live service: `https://panel-158146354492.europe-west2.run.app`

## How it works

1. **Persona generation.** A "casting director" call reads the question,
   identifies the key tensions, and generates three expert personas
   designed to fall on different sides of those tensions.
2. **Round 1.** Each persona answers the question independently, in
   parallel, without seeing anyone else's answer.
3. **Round 2.** Each persona reads the other two answers and responds
   with structured critique: strongest point, weakest point, and one
   question for each colleague. They are instructed not to seek
   consensus.
4. **Summary.** A separate moderator call extracts each persona's final
   position, confidence level, whether they changed their mind (and
   why), the disagreements that persisted, and the questions the panel
   could not settle.

Total: 8 Gemini calls per request (1 casting + 3 round 1 + 3 round 2 +
1 summary). A typical run takes 30 to 90 seconds.

## Redeploy in under 30 minutes

### Prerequisites

- A Google account
- Python 3.12+
- The `gcloud` CLI installed
  (https://cloud.google.com/sdk/docs/install)

### Step 1: Get a Vertex AI API key (5 minutes)

1. Go to https://aistudio.google.com/ and sign in.
2. Click "Get API key" and create a key.
3. Note which Gemini model names are available (the code defaults to
   `gemini-2.5-flash`).

### Step 2: Set up GCP (10 minutes)

1. Go to https://console.cloud.google.com/ and create a new project (or
   use an existing one).
2. Install and configure the CLI:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region europe-west2
```

### Step 3: Clone and deploy (10 minutes)

```bash
git clone https://github.com/esarla01/ExpertPanel.git
cd ExpertPanel
gcloud run deploy panel --source . --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=your-api-key
```

Cloud Run will ask to enable a few services on the first deploy. Say
yes. The build takes a few minutes, then it prints a URL. Test it:

```bash
curl https://YOUR-URL/health
```

If it returns `{"status":"ok"}`, open the URL in a browser to see the
frontend.

### Step 4: Test the panel

From the browser, type a question and click "Run panel." Or use curl:

```bash
curl -X POST https://YOUR-URL/panel \
  -H "Content-Type: application/json" \
  -d '{"question": "Who should get GLP-1 treatment first?"}'
```

## API

### POST /panel

Request body:

```json
{
  "question": "Who should get GLP-1 treatment first?",
  "mode": "dynamic"
}
```

`mode` is optional and defaults to `dynamic`. Options:

| Mode | What it does | When to use |
|---|---|---|
| `dynamic` | Generates personas fitted to the question | Default. Use for any question. |
| `fixed` | Uses pre-defined personas from `personas.yaml` | Comparison baseline for medical questions. |
| `baseline` | Samples one persona three times | Null-diversity control for evaluation. |

Response includes: generated persona definitions, raw round 1 and round
2 text, final positions with confidence and change reasons,
disagreements, and unresolved questions. See `/docs` for the full
schema.

### GET /

Returns the frontend (a single HTML page).

### GET /health

Returns `{"status": "ok"}`.

## Project structure

```
.
├── app.py                FastAPI service with /panel and / endpoints
├── panel/
│   ├── llm.py            Gemini client, retries, caching
│   ├── panel.py          Debate orchestration (casting, rounds, summary)
│   └── schemas.py        Pydantic models for request and response
├── personas.yaml         Fixed personas for fixed/baseline modes
├── static/
│   └── index.html        Frontend
├── evals/
│   └── run_evals.py      Evaluation scripts (placeholder)
├── Dockerfile
├── requirements.txt
└── README.md
```
## Design decisions

I did a brief literature review on multi-agent debate systems and
incorporated findings that were directly relevant to the architecture.

**Independent first round, in parallel.** I run all three personas in
parallel with no visibility of each other's answers. Du et al. (ICML
2024) showed this preserves first-round diversity, and BenchForm (ICLR
2025) found conformity rises the moment agents see each other's output.

**Structured critique in round 2.** Rather than open discussion, each
persona must name the strongest point, weakest point, and one question
for each colleague. I chose this based on Khan et al. (ICML 2024), who
found structured critique surfaces truth more reliably than free-form
debate.

**Anti-convergence prompting.** I instruct personas not to seek
consensus and to lead with disagreement. Smit et al. (ICML 2024) found
that tuning agreeableness is the single strongest lever for debate
quality.

**Dynamic persona generation.** I noticed that fixed personas collapsed
when the question fell outside their domain (all three agreed on a gene
therapy question). So I added a casting step that reads the question and
generates personas designed to disagree on its specific tensions.
ChatEval (Chan et al., ICLR 2024) supports this: identical roles
degraded performance.

**Only two rounds.** I deliberately stopped at two. The Social
Laboratory (arXiv 2510.01295) measured diversity dropping sharply after
each round, and Wynn et al. (2025) showed additional rounds can make
things worse, not better.

**External moderator for summary.** The summary is extracted by a
separate call that saw the debate but did not participate. Khan et al.
(ICML 2024) showed an external judge outperforms self-summary.

**Confidence and unresolved questions.** I ask each persona to state
their confidence and what could change their mind, and the moderator
extracts questions the panel could not settle. This follows
MedAgentAudit (2025), which recommends treating unresolved conflict as
signal rather than smoothing it away.

**Brevity constraint.** I cap responses at 250 words. Without this, the
model hedges and qualifies until the position is buried.

**Baseline mode.** Hu and Collier (ACL 2024) found persona prompts
explain under 10% of output variance. I built a baseline mode (same
persona sampled three times) so I could test whether the diversity I
see is real or just sampling noise.

## Known limitations

The panel is a coverage tool, not a prediction tool. Key limitations
include single-model diversity, no external fact-checking, and
convergence risk in round 2. These are analysed in detail in PART2.md.

## Failure handling

Every Gemini call retries up to 3 times with exponential backoff. If
the summary returns invalid JSON, the model is asked to fix it once; if
that also fails, the response comes back with empty fields and an error
message. If a single persona fails, the other two continue and the
error is logged in the response. If dynamic persona generation fails,
the system falls back to the fixed personas. During development,
responses are cached locally to avoid burning quota; the summary call
is never cached.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Vertex AI express mode API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Which Gemini model to use |
| `CACHE_DIR` | `.cache` | Where to store cached responses |
| `PORT` | `8080` | Port for the server (set by Cloud Run) |

The API key is passed as an environment variable and never appears in
the code or repository.


## AI tools used

- **Claude (Anthropic)** for code scaffolding, prompt design, literature
  review on multi-agent debate, and iterating on the Part 2 note. I
  used Claude to generate initial versions of the codebase and to
  discuss architectural decisions. All design choices and the Part 2
  answers are my own, developed through conversation with Claude as a
  sparring partner.
- **Gemini (Google)** as the underlying model for the panel itself, via
  Vertex AI express mode.

## References

- Du et al., "Improving Factuality and Reasoning in Language Models
  through Multiagent Debate," ICML 2024
- Khan et al., "Debating with More Persuasive LLMs Leads to More
  Truthful Answers," ICML 2024
- Smit et al., "Should we be going MAD? A Look at Multi-Agent Debate
  Strategies for LLMs," ICML 2024
- Chan et al., "ChatEval: Towards Better LLM-based Evaluators through
  Multi-Agent Debate," ICLR 2024
- Hu and Collier, "Quantifying the Persona Effect in LLM Simulations,"
  ACL 2024
- Chen et al., "ReConcile: Round-Table Conference Improves Reasoning via
  Consensus among Diverse LLMs," ACL 2024
- Kim et al., "Correlated Errors in Large Language Models," ICML 2025
- MedAgentAudit, "Auditing Medical Multi-Agent AI Reveals Risks of False
  Consensus," arXiv 2510.10185
- Wynn et al., "Talk Isn't Always Cheap," 2025
- "The Social Laboratory," arXiv 2510.01295
- Liang et al., "Encouraging Divergent Thinking in Large Language Models
  through Multi-Agent Debate," EMNLP 2024