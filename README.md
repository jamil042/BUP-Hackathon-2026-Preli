# GridWise LLM — BUP CSE Fest 2026 Preliminary

LLM-assisted 24-hour campus energy scheduler. Interprets 1-3 natural-language operator notes with Google
Gemini, deterministically guardrails that output, then solves an exact linear program for the cheapest valid
grid/solar/battery schedule.

## Architecture

```
Energy data + operator notes
        -> Gemini interpreter        (app/llm_interpreter.py)
        -> deterministic guardrails  (app/guardrails.py)   -- never trusts raw LLM output,
                                                                falls back to no_op on anything malformed
        -> LP optimizer              (app/optimizer.py)    -- PuLP/CBC, exact global cost optimum
        -> final replay self-check   (app/replay.py)       -- re-validates the plan before responding
        -> JSON response
```

See the full wiring in `app/main.py:optimize_energy`.

Design choices worth knowing:
- The optimizer is an **exact linear program**, not a heuristic — for every feasible scenario it finds the
  true minimum-cost schedule given the applied directives, which maximizes the Optimization Quality score
  (`min(1, organizer_optimal_cost / team_cost)` per hidden case).
- `applies` is **derived from `directive_type`** in `guardrails.py`, never trusted from the LLM's own output —
  `no_op` is always `applies=false`, everything else is always `applies=true`, per the spec's required
  semantics.
- The Gemini prompt is given the scenario's `battery.capacity_kwh` so it can correctly convert
  percentage-of-capacity reserve language (e.g. "keep at least 50% of the battery capacity") into an absolute
  kWh value — without this, percentage-based notes were extracted incorrectly during local testing.
- If Gemini is unavailable or returns unparseable output, the service falls back to marking all notes as
  `no_op` and still returns a fully valid optimized schedule against the base data, rather than failing the
  request (see `LLMUnavailableError` handling in `app/main.py`).

## Model / provider

Google Gemini, model `gemini-flash-lite-latest`, via the `google-generativeai` SDK, JSON response mode.
Requires a `GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey).

Note: `gemini-flash-lite-latest` was chosen over the full `gemini-*-flash` models because the free tier's
generous-enough per-minute quota keeps repeated hidden-judge requests from being rate-limited; a full flash
model on the free tier was observed to hit a 5-requests/minute cap during local testing.

## Repository & live endpoint

- Live judging endpoint: `https://gridwise-llm-kquw.onrender.com`
  - `GET /health` → `{"status":"ok"}`
  - `POST /optimize-energy`

## Local quickstart (clean environment)

```bash
git clone https://github.com/jamil042/BUP-Hackathon-2026-Preli.git
cd BUP-Hackathon-2026-Preli
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
copy .env.example .env        # then edit .env and paste in your real GEMINI_API_KEY
```

Set the environment variable and start the service:

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your-real-key"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
# bash
export GEMINI_API_KEY="your-real-key"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test it

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d "@tests/fixtures/sample_request.json"

python scripts/run_public_samples.py
```

`scripts/run_public_samples.py` replays all 10 cases from `tests/fixtures/public_samples.json` (the official
Public Sample Cases pack) against a running local server and reports directive-type matches plus cost deltas
against the reference optimal cost.

## Docker fallback

```bash
docker build -t gridwise-llm:local .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key-here gridwise-llm:local
curl http://127.0.0.1:8000/health
```

Published image (fill in after pushing):

```bash
docker pull <your-dockerhub-username>/gridwise-llm:prelim
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key-here <your-dockerhub-username>/gridwise-llm:prelim
```

## Dependencies

FastAPI, Uvicorn, Pydantic v2, `google-generativeai`, PuLP (CBC solver, installed via apt in the Docker image),
pytest, httpx. See `requirements.txt` for pinned versions.

## Known limitations

- Single LLM provider (Gemini); no automatic multi-provider failover. Requests that hit a Gemini rate limit
  are retried with exponential backoff (up to 4 attempts, 1s base delay) before the service falls back to
  marking all notes `no_op` and still returning a valid (but interpretation-degraded) schedule rather than
  failing the request.
- The LP optimizer assumes all input scenarios are feasible, per the Problem Statement's guarantee that
  organizer scoring scenarios never require mutually contradictory hard directives.

## Secret handling

No API keys or secrets are committed. `.env` is git-ignored; the Docker image reads `GEMINI_API_KEY` only from
the runtime environment, never baked into the image.
