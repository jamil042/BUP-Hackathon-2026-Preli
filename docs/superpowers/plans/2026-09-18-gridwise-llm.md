# GridWise LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Time-boxed deviation:** This plan was written under a hard 4-hour hackathon deadline (event window 7:00 PM - 11:00 PM). Tasks are grouped coarser than the skill's default 2-5-minute granularity to fit the clock; each task still ends with a real test-then-implement-then-verify-then-commit cycle. Execute inline, not via per-task subagent dispatch — round-trip overhead isn't affordable tonight.

**Goal:** Build, test, containerize, and deploy the GridWise LLM API (`GET /health`, `POST /optimize-energy`) that interprets 1-3 natural-language operator notes with Google Gemini, deterministically guardrails that output, and computes a cost-optimal 24-hour battery/solar/grid schedule via linear programming.

**Architecture:** FastAPI service with five focused modules — `schemas.py` (pydantic contract), `llm_interpreter.py` (single Gemini call → raw directive guesses), `guardrails.py` (deterministic validation, safe fallback to `no_op`, `applies` is derived not trusted), `optimizer.py` (PuLP/CBC linear program — exact global optimum, not a heuristic), `replay.py` (post-solve self-check that replays the plan against every GridWise rule before the response leaves the service, mirroring the "Final Validator" stage in the official pipeline diagram). `main.py` wires them together with controlled error handling (400 for malformed input, 500 generic-only, never a raw stack trace).

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, `google-generativeai` (Gemini `gemini-2.0-flash`, JSON mode), PuLP with bundled CBC solver, pytest, Docker, Render.

**Spec:** Design agreed in conversation (2026-09-18) — canonical source of truth remains `BUP_CSE_FEST_2026_Preliminary_Problem_Statement_GridWise_LLM.pdf` and `BUP_CSE_FEST_2026_Participant_Guide_&_Evaluation_Rubric_GridWise_LLM.pdf` in `F:\BUP_CSE_FEST_2026_Participant_Docs\`.

## Global Constraints

- Endpoint names must match exactly: `GET /health`, `POST /optimize-energy`.
- `hours` arrays inside `structured_adjustment` must be unique ascending ints 0-23.
- Time windows are start-inclusive, end-exclusive (1 PM-3 PM → `[13,14]`).
- `solar_reduction.factor` is the fraction of solar that **remains** (80% reduction → `factor: 0.2`).
- `no_op` is the only directive type allowed with `applies: false`; every other directive uses `applies: true`.
- Numeric tolerance for judge comparisons: 0.01 kWh / 0.01 BDT.
- No secrets committed to the repo; `GEMINI_API_KEY` read from environment only.
- The LLM must genuinely produce the directive interpretation — it is never bypassed by regex/keyword matching, and it is never used only for `plan_summary`.
- `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` in the response must be computed directly from `hourly_plan`, never independently.
- Service must never crash or invent an unsupported directive type on bad LLM output; malformed input returns a controlled error, not a 500 with a stack trace.

---

## Task 1: Project scaffold, schemas, and `/health`

**Files:**
- Create: `F:\GridWise-LLM\requirements.txt`
- Create: `F:\GridWise-LLM\app\__init__.py`
- Create: `F:\GridWise-LLM\app\schemas.py`
- Create: `F:\GridWise-LLM\app\main.py`
- Create: `F:\GridWise-LLM\tests\__init__.py`
- Test: `F:\GridWise-LLM\tests\test_health.py`
- Create: `F:\GridWise-LLM\.env.example`
- Create: `F:\GridWise-LLM\.gitignore`

**Interfaces:**
- Produces: `app.schemas.OptimizeRequest`, `app.schemas.OptimizeResponse`, `app.schemas.HourEntry`, `app.schemas.Battery`, `app.schemas.DirectiveInterpretation`, `app.schemas.HourlyPlanEntry` (pydantic models matching the Problem Statement §7/§10 field-for-field).
- Produces: FastAPI app instance `app.main.app` with `GET /health`.

- [ ] **Step 1: Write requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
google-generativeai==0.8.3
pulp==2.9.0
python-dotenv==1.0.1
pytest==8.3.3
httpx==0.27.2
```

- [ ] **Step 2: Write the schemas**

```python
# app/schemas.py
from typing import Literal, Optional
from pydantic import BaseModel, Field, conlist

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

class HourEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)

class Battery(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: conlist(str, min_length=1, max_length=3)
    hours: conlist(HourEntry, min_length=24, max_length=24)
    battery: Battery

class StructuredAdjustment(BaseModel):
    hours: Optional[list[int]] = None
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None

class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[StructuredAdjustment]
    explanation: str

class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float

class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
```

- [ ] **Step 3: Write main.py with only `/health`**

```python
# app/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI(title="GridWise LLM")

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return JSONResponse(status_code=400, content={"error": "malformed_or_invalid_request", "detail": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "internal_error"})

@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Write the failing test**

```python
# tests/test_health.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 5: Install deps and run test**

Run: `cd F:\GridWise-LLM; python -m venv .venv; .venv\Scripts\pip install -r requirements.txt; .venv\Scripts\pytest tests/test_health.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: .env.example and .gitignore**

```
# .env.example
GEMINI_API_KEY=your-gemini-api-key-here
```

```
# .gitignore
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/ tests/ .env.example .gitignore
git commit -m "feat: project scaffold, pydantic schemas, /health endpoint"
```

---

## Task 2: Guardrails module (deterministic LLM-output validation)

**Files:**
- Create: `F:\GridWise-LLM\app\guardrails.py`
- Test: `F:\GridWise-LLM\tests\test_guardrails.py`

**Interfaces:**
- Consumes: `app.schemas.StructuredAdjustment`, `app.schemas.DirectiveInterpretation`
- Produces: `app.guardrails.validate(raw_directives: list[dict], notes: list[str], battery_capacity_kwh: float) -> list[DirectiveInterpretation]` — always returns exactly `len(notes)` entries, one per `note_index` 0..N-1, in order, never raises.
- Produces: `app.guardrails.fallback_all_no_op(notes: list[str], reason: str) -> list[DirectiveInterpretation]`

- [ ] **Step 1: Write failing tests covering every guardrail rule**

```python
# tests/test_guardrails.py
from app.guardrails import validate, fallback_all_no_op

NOTES = ["Solar drops to 20% from 1pm to 3pm.", "The cafeteria menu changes."]

def test_valid_solar_reduction_passes_through():
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [13, 14], "factor": 0.2}, "explanation": "x"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "irrelevant"},
    ]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert len(result) == 2
    assert result[0].directive_type == "solar_reduction"
    assert result[0].applies is True
    assert result[0].structured_adjustment.hours == [13, 14]
    assert result[1].directive_type == "no_op"
    assert result[1].applies is False
    assert result[1].structured_adjustment is None

def test_unsupported_directive_type_falls_back_to_no_op():
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "shutdown_campus",
         "structured_adjustment": {"hours": [1]}, "explanation": "x"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "x"},
    ]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"
    assert result[0].applies is False

def test_missing_note_index_defaults_to_no_op():
    raw = [{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [13, 14], "factor": 0.2}, "explanation": "x"}]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert len(result) == 2
    assert result[1].note_index == 1
    assert result[1].directive_type == "no_op"

def test_unordered_and_unsorted_hours_are_deduped_and_sorted():
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [14, 13, 13], "factor": 0.2}, "explanation": "x"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "x"},
    ]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert result[0].structured_adjustment.hours == [13, 14]

def test_out_of_range_factor_falls_back_to_no_op():
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [13], "factor": 1.5}, "explanation": "x"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "x"},
    ]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"

def test_reserve_exceeding_capacity_falls_back_to_no_op():
    raw = [
        {"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [18], "minimum_energy_kwh": 9999}, "explanation": "x"},
        {"note_index": 1, "applies": False, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "x"},
    ]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert result[0].directive_type == "no_op"

def test_applies_is_derived_not_trusted():
    raw = [
        {"note_index": 0, "applies": False, "directive_type": "no_charge_window",
         "structured_adjustment": {"hours": [2, 3]}, "explanation": "x"},
        {"note_index": 1, "applies": True, "directive_type": "no_op",
         "structured_adjustment": None, "explanation": "x"},
    ]
    result = validate(raw, NOTES, battery_capacity_kwh=500)
    assert result[0].applies is True
    assert result[1].applies is False

def test_fallback_all_no_op_covers_every_note():
    result = fallback_all_no_op(NOTES, reason="llm_unavailable")
    assert len(result) == 2
    assert all(r.directive_type == "no_op" and r.applies is False for r in result)
```

- [ ] **Step 2: Run tests, verify all fail**

Run: `.venv\Scripts\pytest tests/test_guardrails.py -v`
Expected: FAIL (ModuleNotFoundError: app.guardrails)

- [ ] **Step 3: Implement guardrails.py**

```python
# app/guardrails.py
from app.schemas import DirectiveInterpretation, StructuredAdjustment

ALLOWED_TYPES = {
    "solar_reduction", "minimum_battery_reserve", "no_charge_window",
    "no_discharge_window", "max_grid_window", "no_op",
}

def _clean_hours(raw_hours) -> list[int] | None:
    if not isinstance(raw_hours, list) or not raw_hours:
        return None
    try:
        hours = sorted({int(h) for h in raw_hours})
    except (TypeError, ValueError):
        return None
    if any(h < 0 or h > 23 for h in hours):
        return None
    return hours

def _no_op(note_index: int, explanation: str) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index, applies=False, directive_type="no_op",
        structured_adjustment=None, explanation=explanation,
    )

def _build_directive(note_index: int, entry: dict, battery_capacity_kwh: float) -> DirectiveInterpretation:
    dtype = entry.get("directive_type")
    explanation = str(entry.get("explanation") or "")

    if dtype == "no_op" or dtype not in ALLOWED_TYPES:
        return _no_op(note_index, explanation or "No applicable energy directive found in this note.")

    adj = entry.get("structured_adjustment") or {}
    hours = _clean_hours(adj.get("hours"))
    if hours is None:
        return _no_op(note_index, "Malformed hours in model output; defaulted to no_op.")

    if dtype == "solar_reduction":
        factor = adj.get("factor")
        if not isinstance(factor, (int, float)) or not (0 <= factor <= 1):
            return _no_op(note_index, "Invalid solar_reduction factor; defaulted to no_op.")
        return DirectiveInterpretation(
            note_index=note_index, applies=True, directive_type=dtype,
            structured_adjustment=StructuredAdjustment(hours=hours, factor=float(factor)),
            explanation=explanation or "Solar output reduced for the stated window.",
        )

    if dtype == "minimum_battery_reserve":
        reserve = adj.get("minimum_energy_kwh")
        if not isinstance(reserve, (int, float)) or reserve < 0 or reserve > battery_capacity_kwh:
            return _no_op(note_index, "Invalid battery reserve value; defaulted to no_op.")
        return DirectiveInterpretation(
            note_index=note_index, applies=True, directive_type=dtype,
            structured_adjustment=StructuredAdjustment(hours=hours, minimum_energy_kwh=float(reserve)),
            explanation=explanation or "Minimum battery reserve enforced for the stated window.",
        )

    if dtype in ("no_charge_window", "no_discharge_window"):
        return DirectiveInterpretation(
            note_index=note_index, applies=True, directive_type=dtype,
            structured_adjustment=StructuredAdjustment(hours=hours),
            explanation=explanation or "Battery action disabled for the stated window.",
        )

    if dtype == "max_grid_window":
        cap = adj.get("max_grid_kwh")
        if not isinstance(cap, (int, float)) or cap < 0:
            return _no_op(note_index, "Invalid max_grid_kwh value; defaulted to no_op.")
        return DirectiveInterpretation(
            note_index=note_index, applies=True, directive_type=dtype,
            structured_adjustment=StructuredAdjustment(hours=hours, max_grid_kwh=float(cap)),
            explanation=explanation or "Grid import capped for the stated window.",
        )

    return _no_op(note_index, "Unrecognized directive shape; defaulted to no_op.")

def validate(raw_directives: list[dict], notes: list[str], battery_capacity_kwh: float) -> list[DirectiveInterpretation]:
    by_index: dict[int, dict] = {}
    if isinstance(raw_directives, list):
        for entry in raw_directives:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("note_index")
            if isinstance(idx, int) and 0 <= idx < len(notes) and idx not in by_index:
                by_index[idx] = entry

    results = []
    for i in range(len(notes)):
        entry = by_index.get(i)
        if entry is None:
            results.append(_no_op(i, "No interpretation returned for this note; defaulted to no_op."))
        else:
            results.append(_build_directive(i, entry, battery_capacity_kwh))
    return results

def fallback_all_no_op(notes: list[str], reason: str) -> list[DirectiveInterpretation]:
    return [_no_op(i, f"Interpretation unavailable ({reason}); defaulted to no_op.") for i in range(len(notes))]
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `.venv\Scripts\pytest tests/test_guardrails.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add app/guardrails.py tests/test_guardrails.py
git commit -m "feat: deterministic guardrail validation for LLM directive output"
```

---

## Task 3: Optimizer module (exact LP solve for the 24-hour schedule)

**Files:**
- Create: `F:\GridWise-LLM\app\optimizer.py`
- Test: `F:\GridWise-LLM\tests\test_optimizer.py`

**Interfaces:**
- Consumes: `app.schemas.HourEntry`, `app.schemas.Battery`, `app.schemas.DirectiveInterpretation`
- Produces: `app.optimizer.solve(hours: list[HourEntry], battery: Battery, directives: list[DirectiveInterpretation]) -> list[HourlyPlanEntry]`

This is a linear program, so PuLP/CBC returns the true global-optimal cost, not an approximation — this is what maximizes the Optimization Quality score (10 pts = `10 x organizer_optimal_cost / team_cost`, capped at 1.0 per case; an exact LP ties or beats any heuristic).

**LP formulation:**
- Variables per hour h=0..23: `grid[h] >= 0`, `solar_used[h] >= 0`, `charge[h] >= 0`, `discharge[h] >= 0`, `batt[h]` (energy after hour h).
- `effective_solar[h] = solar_kwh[h] * factor` if a `solar_reduction` directive covers h, else `solar_kwh[h]`. `solar_used[h] <= effective_solar[h]`.
- `charge[h] <= max_charge_kwh_per_hour` (0 if `no_charge_window` covers h). `discharge[h] <= max_discharge_kwh_per_hour` (0 if `no_discharge_window` covers h).
- `grid[h] <= max_grid_kwh` if a `max_grid_window` directive covers h.
- Battery recurrence: `batt[h] == (batt[h-1] if h>0 else initial_energy_kwh) + charge[h] - discharge[h]`.
- Bounds: `batt[h] >= max(minimum_energy_kwh, directive_reserve[h])`, `batt[h] <= capacity_kwh`.
- End-of-day neutrality: `batt[23] == initial_energy_kwh`.
- Energy balance: `grid[h] + solar_used[h] + discharge[h] == demand_kwh[h] + charge[h]`.
- Objective: `minimize sum(grid[h] * tariff_bdt_per_kwh[h])`.
- After solving, `battery_action`/`battery_kwh` per hour come from `net = charge[h] - discharge[h]`: `net > 1e-6` → charge; `net < -1e-6` → discharge; else idle with `battery_kwh = 0`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_optimizer.py
from app.schemas import HourEntry, Battery, DirectiveInterpretation, StructuredAdjustment
from app.optimizer import solve

def _flat_hours(demand=100.0, solar=0.0, tariff=10.0):
    return [HourEntry(hour=h, demand_kwh=demand, solar_kwh=solar, tariff_bdt_per_kwh=tariff) for h in range(24)]

def _battery(**overrides):
    defaults = dict(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=20,
                     max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    defaults.update(overrides)
    return Battery(**defaults)

def test_no_directives_flat_tariff_battery_stays_idle_and_meets_demand():
    plan = solve(_flat_hours(), _battery(), directives=[])
    assert len(plan) == 24
    total_grid = sum(e.grid_kwh for e in plan)
    assert abs(total_grid - 2400.0) < 0.1
    assert abs(plan[-1].battery_energy_after_kwh - 100.0) < 0.01

def test_cheap_then_expensive_tariff_shifts_battery_energy():
    hours = [HourEntry(hour=h, demand_kwh=100.0, solar_kwh=0.0,
                        tariff_bdt_per_kwh=(5.0 if h < 12 else 20.0)) for h in range(24)]
    plan = solve(hours, _battery(), directives=[])
    morning_charge = sum(e.battery_kwh for e in plan[:12] if e.battery_action == "charge")
    evening_discharge = sum(e.battery_kwh for e in plan[12:] if e.battery_action == "discharge")
    assert morning_charge > 0
    assert evening_discharge > 0
    assert abs(plan[-1].battery_energy_after_kwh - 100.0) < 0.01

def test_no_charge_window_forces_zero_charge_in_window():
    directive = DirectiveInterpretation(
        note_index=0, applies=True, directive_type="no_charge_window",
        structured_adjustment=StructuredAdjustment(hours=[2, 3, 4]), explanation="x")
    hours = [HourEntry(hour=h, demand_kwh=100.0, solar_kwh=0.0,
                        tariff_bdt_per_kwh=(2.0 if h < 6 else 15.0)) for h in range(24)]
    plan = solve(hours, _battery(), directives=[directive])
    for h in (2, 3, 4):
        assert plan[h].battery_action != "charge"

def test_minimum_battery_reserve_is_respected():
    directive = DirectiveInterpretation(
        note_index=0, applies=True, directive_type="minimum_battery_reserve",
        structured_adjustment=StructuredAdjustment(hours=[18, 19, 20], minimum_energy_kwh=120),
        explanation="x")
    hours = [HourEntry(hour=h, demand_kwh=100.0, solar_kwh=0.0, tariff_bdt_per_kwh=10.0) for h in range(24)]
    plan = solve(hours, _battery(initial_energy_kwh=150), directives=[directive])
    for h in (18, 19, 20):
        assert plan[h].battery_energy_after_kwh >= 120 - 0.01

def test_energy_balance_holds_every_hour():
    hours = [HourEntry(hour=h, demand_kwh=100.0 + h, solar_kwh=float(h * 3), tariff_bdt_per_kwh=10.0) for h in range(24)]
    plan = solve(hours, _battery(), directives=[])
    for h, entry in enumerate(plan):
        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action == "discharge" else 0.0
        lhs = entry.grid_kwh + entry.solar_used_kwh + discharge
        rhs = hours[h].demand_kwh + charge
        assert abs(lhs - rhs) < 0.02
```

- [ ] **Step 2: Run tests, verify all fail**

Run: `.venv\Scripts\pytest tests/test_optimizer.py -v`
Expected: FAIL (ModuleNotFoundError: app.optimizer)

- [ ] **Step 3: Implement optimizer.py**

```python
# app/optimizer.py
import pulp
from app.schemas import HourEntry, Battery, DirectiveInterpretation, HourlyPlanEntry

TOL = 1e-6

def _effective_solar(hours: list[HourEntry], directives: list[DirectiveInterpretation]) -> list[float]:
    solar = [h.solar_kwh for h in hours]
    for d in directives:
        if d.applies and d.directive_type == "solar_reduction":
            for h in d.structured_adjustment.hours:
                solar[h] = hours[h].solar_kwh * d.structured_adjustment.factor
    return solar

def _reserve_floor(hours: list[HourEntry], battery: Battery, directives: list[DirectiveInterpretation]) -> list[float]:
    floor = [battery.minimum_energy_kwh] * 24
    for d in directives:
        if d.applies and d.directive_type == "minimum_battery_reserve":
            for h in d.structured_adjustment.hours:
                floor[h] = max(floor[h], d.structured_adjustment.minimum_energy_kwh)
    return floor

def _blocked_hours(directives: list[DirectiveInterpretation], directive_type: str) -> set[int]:
    blocked = set()
    for d in directives:
        if d.applies and d.directive_type == directive_type:
            blocked.update(d.structured_adjustment.hours)
    return blocked

def _grid_caps(directives: list[DirectiveInterpretation]) -> dict[int, float]:
    caps: dict[int, float] = {}
    for d in directives:
        if d.applies and d.directive_type == "max_grid_window":
            for h in d.structured_adjustment.hours:
                caps[h] = min(caps.get(h, d.structured_adjustment.max_grid_kwh), d.structured_adjustment.max_grid_kwh)
    return caps

def solve(hours: list[HourEntry], battery: Battery, directives: list[DirectiveInterpretation]) -> list[HourlyPlanEntry]:
    effective_solar = _effective_solar(hours, directives)
    reserve_floor = _reserve_floor(hours, battery, directives)
    no_charge = _blocked_hours(directives, "no_charge_window")
    no_discharge = _blocked_hours(directives, "no_discharge_window")
    grid_caps = _grid_caps(directives)

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)

    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=grid_caps.get(h)) for h in range(24)}
    solar_used = {h: pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=effective_solar[h]) for h in range(24)}
    charge = {h: pulp.LpVariable(f"chg_{h}", lowBound=0, upBound=(0 if h in no_charge else battery.max_charge_kwh_per_hour)) for h in range(24)}
    discharge = {h: pulp.LpVariable(f"dis_{h}", lowBound=0, upBound=(0 if h in no_discharge else battery.max_discharge_kwh_per_hour)) for h in range(24)}
    batt = {h: pulp.LpVariable(f"batt_{h}", lowBound=reserve_floor[h], upBound=battery.capacity_kwh) for h in range(24)}

    prob += pulp.lpSum(grid[h] * hours[h].tariff_bdt_per_kwh for h in range(24))

    for h in range(24):
        prob += grid[h] + solar_used[h] + discharge[h] == hours[h].demand_kwh + charge[h]
        prev = battery.initial_energy_kwh if h == 0 else batt[h - 1]
        prob += batt[h] == prev + charge[h] - discharge[h]

    prob += batt[23] == battery.initial_energy_kwh

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"optimizer_infeasible: {pulp.LpStatus[status]}")

    plan = []
    for h in range(24):
        net = pulp.value(charge[h]) - pulp.value(discharge[h])
        if net > TOL:
            action, magnitude = "charge", net
        elif net < -TOL:
            action, magnitude = "discharge", -net
        else:
            action, magnitude = "idle", 0.0
        plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=max(0.0, round(pulp.value(grid[h]), 6)),
            solar_used_kwh=max(0.0, round(pulp.value(solar_used[h]), 6)),
            battery_action=action,
            battery_kwh=round(magnitude, 6),
            battery_energy_after_kwh=round(pulp.value(batt[h]), 6),
        ))
    return plan
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `.venv\Scripts\pytest tests/test_optimizer.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/optimizer.py tests/test_optimizer.py
git commit -m "feat: exact LP optimizer for 24-hour battery/solar/grid schedule"
```

---

## Task 4: Final replay validator (defense-in-depth self-check)

**Files:**
- Create: `F:\GridWise-LLM\app\replay.py`
- Test: `F:\GridWise-LLM\tests\test_replay.py`

**Interfaces:**
- Consumes: `app.schemas.HourEntry`, `app.schemas.Battery`, `app.schemas.DirectiveInterpretation`, `app.schemas.HourlyPlanEntry`
- Produces: `app.replay.replay_and_check(hours, battery, directives, plan) -> list[str]` — returns a list of human-readable violation strings; empty list means the plan is fully valid. Used by `main.py` to log (never to silently mutate) any anomaly before responding, mirroring the official "Final Validator" pipeline stage.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_replay.py
from app.schemas import HourEntry, Battery, HourlyPlanEntry
from app.replay import replay_and_check

def _hours():
    return [HourEntry(hour=h, demand_kwh=100.0, solar_kwh=0.0, tariff_bdt_per_kwh=10.0) for h in range(24)]

def _battery():
    return Battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=20,
                    max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)

def _idle_plan_meeting_demand():
    return [HourlyPlanEntry(hour=h, grid_kwh=100.0, solar_used_kwh=0.0, battery_action="idle",
                             battery_kwh=0.0, battery_energy_after_kwh=100.0) for h in range(24)]

def test_valid_plan_has_no_violations():
    violations = replay_and_check(_hours(), _battery(), [], _idle_plan_meeting_demand())
    assert violations == []

def test_energy_balance_violation_detected():
    plan = _idle_plan_meeting_demand()
    plan[0].grid_kwh = 50.0
    violations = replay_and_check(_hours(), _battery(), [], plan)
    assert any("energy balance" in v for v in violations)

def test_end_of_day_neutrality_violation_detected():
    plan = _idle_plan_meeting_demand()
    plan[-1].battery_energy_after_kwh = 150.0
    violations = replay_and_check(_hours(), _battery(), [], plan)
    assert any("end-of-day" in v for v in violations)

def test_battery_below_minimum_detected():
    plan = _idle_plan_meeting_demand()
    plan[5].battery_energy_after_kwh = 5.0
    violations = replay_and_check(_hours(), _battery(), [], plan)
    assert any("minimum" in v for v in violations)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `.venv\Scripts\pytest tests/test_replay.py -v`
Expected: FAIL (ModuleNotFoundError: app.replay)

- [ ] **Step 3: Implement replay.py**

```python
# app/replay.py
from app.schemas import HourEntry, Battery, DirectiveInterpretation, HourlyPlanEntry
from app.optimizer import _effective_solar, _reserve_floor, _blocked_hours, _grid_caps

TOL = 0.02

def replay_and_check(hours: list[HourEntry], battery: Battery,
                      directives: list[DirectiveInterpretation], plan: list[HourlyPlanEntry]) -> list[str]:
    violations: list[str] = []
    if len(plan) != 24:
        return [f"hourly_plan must contain exactly 24 entries, got {len(plan)}"]

    effective_solar = _effective_solar(hours, directives)
    reserve_floor = _reserve_floor(hours, battery, directives)
    no_charge = _blocked_hours(directives, "no_charge_window")
    no_discharge = _blocked_hours(directives, "no_discharge_window")
    grid_caps = _grid_caps(directives)

    prev_energy = battery.initial_energy_kwh
    for h, entry in enumerate(plan):
        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action == "discharge" else 0.0

        if abs((entry.grid_kwh + entry.solar_used_kwh + discharge) - (hours[h].demand_kwh + charge)) > TOL:
            violations.append(f"hour {h}: energy balance does not hold")
        if entry.solar_used_kwh > effective_solar[h] + TOL:
            violations.append(f"hour {h}: solar_used_kwh exceeds effective solar")
        if entry.grid_kwh < -TOL:
            violations.append(f"hour {h}: negative grid_kwh")
        if h in grid_caps and entry.grid_kwh > grid_caps[h] + TOL:
            violations.append(f"hour {h}: max_grid_window exceeded")
        if h in no_charge and charge > TOL:
            violations.append(f"hour {h}: charge occurred during no_charge_window")
        if h in no_discharge and discharge > TOL:
            violations.append(f"hour {h}: discharge occurred during no_discharge_window")
        if charge > battery.max_charge_kwh_per_hour + TOL:
            violations.append(f"hour {h}: charge exceeds max_charge_kwh_per_hour")
        if discharge > battery.max_discharge_kwh_per_hour + TOL:
            violations.append(f"hour {h}: discharge exceeds max_discharge_kwh_per_hour")

        expected_energy = prev_energy + charge - discharge
        if abs(entry.battery_energy_after_kwh - expected_energy) > TOL:
            violations.append(f"hour {h}: battery_energy_after_kwh inconsistent with battery action")
        if entry.battery_energy_after_kwh < reserve_floor[h] - TOL:
            violations.append(f"hour {h}: battery energy below required minimum reserve")
        if entry.battery_energy_after_kwh > battery.capacity_kwh + TOL:
            violations.append(f"hour {h}: battery energy exceeds capacity")

        prev_energy = entry.battery_energy_after_kwh

    if abs(plan[-1].battery_energy_after_kwh - battery.initial_energy_kwh) > TOL:
        violations.append("end-of-day battery energy does not equal initial_energy_kwh")

    return violations
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `.venv\Scripts\pytest tests/test_replay.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/replay.py tests/test_replay.py
git commit -m "feat: final replay validator as defense-in-depth self-check"
```

---

## Task 5: LLM interpreter (Gemini)

**Files:**
- Create: `F:\GridWise-LLM\app\prompts.py`
- Create: `F:\GridWise-LLM\app\llm_interpreter.py`
- Test: `F:\GridWise-LLM\tests\test_llm_interpreter.py`

**Interfaces:**
- Produces: `app.llm_interpreter.interpret_notes(notes: list[str], model_client=None) -> list[dict]` — returns raw (untrusted) directive dicts straight from Gemini's JSON output, to be passed into `app.guardrails.validate`. Never raises past its own boundary for API errors — raises only `app.llm_interpreter.LLMUnavailableError` so `main.py` can trigger the guardrails fallback path.
- Consumes (test only): an injectable fake client so parsing logic is tested without a live API key.

- [ ] **Step 1: Write the prompt template**

```python
# app/prompts.py
SYSTEM_PROMPT = """You are the operator-note interpreter for BUP's GridWise campus energy system.
You will be given a numbered list of short natural-language notes from a campus operator, all referring
to the same upcoming 24-hour energy schedule (hours 0-23).

For EACH note, decide whether it describes a change to today's energy schedule, and if so, convert it into
EXACTLY ONE of these six directive types:

- solar_reduction: usable solar is reduced during specific hours.
  structured_adjustment: {"hours": [int,...], "factor": number}  // factor = fraction of solar that REMAINS.
    Example: an 80% reduction means factor = 0.2.
- minimum_battery_reserve: battery energy must stay at or above a level during specific hours.
  structured_adjustment: {"hours": [int,...], "minimum_energy_kwh": number}
- no_charge_window: battery charging is unavailable during specific hours.
  structured_adjustment: {"hours": [int,...]}
- no_discharge_window: battery discharging is unavailable during specific hours.
  structured_adjustment: {"hours": [int,...]}
- max_grid_window: grid import may not exceed a stated amount during specific hours.
  structured_adjustment: {"hours": [int,...], "max_grid_kwh": number}
- no_op: the note does NOT affect today's 24-hour energy schedule (distractor, unrelated topic, or already-default behavior).
  structured_adjustment: null

Rules:
- Time windows are start-inclusive, end-exclusive. "1 PM to 3 PM" means hours [13, 14] (NOT 15).
- "hours" must be unique integers from 0 to 23, ascending.
- If a note is ambiguous or does not clearly map to one of the five real directive types, use no_op. Never invent a new directive type.
- Return EXACTLY one object per input note, in the same order, using its 0-based index as note_index.

Respond with ONLY a JSON array (no prose, no markdown fences) where each element has this shape:
{"note_index": int, "directive_type": string, "structured_adjustment": object or null, "explanation": string}
"""

def build_user_prompt(notes: list[str]) -> str:
    numbered = "\n".join(f"{i}: {note}" for i, note in enumerate(notes))
    return f"Operator notes:\n{numbered}\n\nReturn the JSON array now."
```

- [ ] **Step 2: Write failing tests using an injectable fake client**

```python
# tests/test_llm_interpreter.py
import json
import pytest
from app.llm_interpreter import interpret_notes, LLMUnavailableError

class FakeResponse:
    def __init__(self, text):
        self.text = text

class FakeModel:
    def __init__(self, reply_text=None, raise_exc=None):
        self.reply_text = reply_text
        self.raise_exc = raise_exc

    def generate_content(self, *args, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        return FakeResponse(self.reply_text)

def test_parses_valid_json_array_from_model():
    reply = json.dumps([
        {"note_index": 0, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [13, 14], "factor": 0.2}, "explanation": "x"}
    ])
    result = interpret_notes(["Solar drops to 20% 1-3pm."], model_client=FakeModel(reply_text=reply))
    assert result[0]["directive_type"] == "solar_reduction"

def test_strips_markdown_fences_if_present():
    reply = "```json\n" + json.dumps([{"note_index": 0, "directive_type": "no_op",
                                        "structured_adjustment": None, "explanation": "x"}]) + "\n```"
    result = interpret_notes(["Unrelated note."], model_client=FakeModel(reply_text=reply))
    assert result[0]["directive_type"] == "no_op"

def test_raises_llm_unavailable_on_client_exception():
    with pytest.raises(LLMUnavailableError):
        interpret_notes(["Any note."], model_client=FakeModel(raise_exc=RuntimeError("api down")))

def test_raises_llm_unavailable_on_unparseable_output():
    with pytest.raises(LLMUnavailableError):
        interpret_notes(["Any note."], model_client=FakeModel(reply_text="not json at all"))
```

- [ ] **Step 3: Run tests, verify failure**

Run: `.venv\Scripts\pytest tests/test_llm_interpreter.py -v`
Expected: FAIL (ModuleNotFoundError: app.llm_interpreter)

- [ ] **Step 4: Implement llm_interpreter.py**

```python
# app/llm_interpreter.py
import json
import os
import re
from app.prompts import SYSTEM_PROMPT, build_user_prompt

class LLMUnavailableError(Exception):
    pass

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()

def _get_default_client():
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailableError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )

def interpret_notes(notes: list[str], model_client=None) -> list[dict]:
    client = model_client if model_client is not None else _get_default_client()
    try:
        response = client.generate_content(build_user_prompt(notes))
        raw_text = _strip_fences(response.text)
        parsed = json.loads(raw_text)
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(str(exc)) from exc

    if not isinstance(parsed, list):
        raise LLMUnavailableError("model did not return a JSON array")
    return parsed
```

- [ ] **Step 5: Run tests, verify all pass**

Run: `.venv\Scripts\pytest tests/test_llm_interpreter.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add app/prompts.py app/llm_interpreter.py tests/test_llm_interpreter.py
git commit -m "feat: Gemini-backed operator-note interpreter with fake-client test coverage"
```

---

## Task 6: Wire the full pipeline into `POST /optimize-energy`

**Files:**
- Modify: `F:\GridWise-LLM\app\main.py`
- Create: `F:\GridWise-LLM\tests\fixtures\public_samples.json`
- Test: `F:\GridWise-LLM\tests\test_optimize_energy.py`

**Interfaces:**
- Consumes: every module produced in Tasks 1-5.
- Produces: fully wired `POST /optimize-energy` returning `app.schemas.OptimizeResponse`.

- [ ] **Step 1: Copy the public sample cases into the repo as a test fixture**

Run: `Copy-Item "F:\BUP_CSE_FEST_2026_Participant_Docs\BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json" "F:\GridWise-LLM\tests\fixtures\public_samples.json"`

- [ ] **Step 2: Write failing integration tests**

```python
# tests/test_optimize_energy.py
import json
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
FIXTURES = json.loads(Path(__file__).parent.joinpath("fixtures/public_samples.json").read_text())

def test_health_still_ok():
    assert client.get("/health").status_code == 200

def test_malformed_json_returns_400():
    resp = client.post("/optimize-energy", data="{not valid json", headers={"content-type": "application/json"})
    assert resp.status_code == 400

def test_missing_required_field_returns_400():
    resp = client.post("/optimize-energy", json={"scenario_id": "X"})
    assert resp.status_code == 400

def test_llm_failure_falls_back_to_no_op_and_still_returns_valid_plan():
    case = FIXTURES["cases"][1]  # SAMPLE-02, single relevant note
    with patch("app.main.interpret_notes", side_effect=Exception("simulated outage")):
        resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hourly_plan"]) == 24
    assert all(d["directive_type"] == "no_op" for d in body["directive_interpretation"])

def test_full_pipeline_with_mocked_llm_matches_expected_directive_types():
    case = FIXTURES["cases"][0]  # SAMPLE-01, solar_reduction + distractor
    expected = case["expected_output"]["directive_interpretation"]
    fake_raw = [
        {"note_index": e["note_index"], "directive_type": e["directive_type"],
         "structured_adjustment": e["structured_adjustment"], "explanation": e["explanation"]}
        for e in expected
    ]
    with patch("app.main.interpret_notes", return_value=fake_raw):
        resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["scenario_id"] == case["input"]["scenario_id"]
    got_types = sorted(d["directive_type"] for d in body["directive_interpretation"])
    want_types = sorted(d["directive_type"] for d in expected)
    assert got_types == want_types
    assert abs(body["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"]) < 1.0
    assert abs(body["total_grid_kwh"] - sum(h["grid_kwh"] for h in body["hourly_plan"])) < 0.02
    assert body["peak_grid_kwh"] == max(h["grid_kwh"] for h in body["hourly_plan"])
```

- [ ] **Step 3: Run tests, verify failure**

Run: `.venv\Scripts\pytest tests/test_optimize_energy.py -v`
Expected: FAIL (404 on /optimize-energy, ImportError on interpret_notes patch target)

- [ ] **Step 4: Implement the endpoint in main.py**

```python
# app/main.py  (replace file contents)
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.schemas import OptimizeRequest, OptimizeResponse, DirectiveInterpretation
from app.llm_interpreter import interpret_notes, LLMUnavailableError
from app.guardrails import validate, fallback_all_no_op
from app.optimizer import solve
from app.replay import replay_and_check

logger = logging.getLogger("gridwise")
app = FastAPI(title="GridWise LLM")

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return JSONResponse(status_code=400, content={"error": "malformed_or_invalid_request", "detail": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_handler(request, exc):
    logger.exception("unhandled_error")
    return JSONResponse(status_code=500, content={"error": "internal_error"})

@app.get("/health")
async def health():
    return {"status": "ok"}

def _summarize(scenario_id: str, directives: list[DirectiveInterpretation], total_cost_bdt: float) -> str:
    applied = [d.directive_type for d in directives if d.applies]
    if applied:
        return f"Applied {', '.join(applied)} for scenario {scenario_id}; total grid cost {total_cost_bdt:.2f} BDT."
    return f"No operator directives applied for scenario {scenario_id}; total grid cost {total_cost_bdt:.2f} BDT."

@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: OptimizeRequest):
    try:
        raw_directives = interpret_notes(request.operator_notes)
    except (LLMUnavailableError, Exception) as exc:
        logger.warning("llm_interpretation_failed: %s", exc)
        directives = fallback_all_no_op(request.operator_notes, reason="llm_unavailable")
    else:
        directives = validate(raw_directives, request.operator_notes, request.battery.capacity_kwh)

    hourly_plan = solve(request.hours, request.battery, directives)

    violations = replay_and_check(request.hours, request.battery, directives, hourly_plan)
    if violations:
        logger.error("replay_validation_failed scenario=%s violations=%s", request.scenario_id, violations)

    total_grid_kwh = round(sum(e.grid_kwh for e in hourly_plan), 6)
    total_cost_bdt = round(sum(e.grid_kwh * h.tariff_bdt_per_kwh for e, h in zip(hourly_plan, request.hours)), 6)
    peak_grid_kwh = round(max(e.grid_kwh for e in hourly_plan), 6)

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid_kwh,
        total_cost_bdt=total_cost_bdt,
        peak_grid_kwh=peak_grid_kwh,
        plan_summary=_summarize(request.scenario_id, directives, total_cost_bdt),
    )
```

- [ ] **Step 5: Run tests, verify all pass**

Run: `.venv\Scripts\pytest tests/ -v`
Expected: PASS (all tests across all files)

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_optimize_energy.py tests/fixtures/public_samples.json
git commit -m "feat: wire LLM interpreter, guardrails, optimizer, and replay validator into POST /optimize-energy"
```

---

## Task 7: Full public-sample regression run against a live local server

**Files:**
- Create: `F:\GridWise-LLM\scripts\run_public_samples.py`

**Interfaces:**
- Consumes: a running local server on `http://127.0.0.1:8000` and `tests/fixtures/public_samples.json`.
- Produces: console report of pass/fail per sample case (interpretation directive types + replay validity), used as a manual gate before deploying.

- [ ] **Step 1: Write the script**

```python
# scripts/run_public_samples.py
import json
import sys
from pathlib import Path
import httpx

BASE_URL = "http://127.0.0.1:8000"
FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "public_samples.json"

def main():
    data = json.loads(FIXTURE.read_text())
    failures = 0
    for case in data["cases"]:
        resp = httpx.post(f"{BASE_URL}/optimize-energy", json=case["input"], timeout=30.0)
        if resp.status_code != 200:
            print(f"[FAIL] {case['id']}: HTTP {resp.status_code} {resp.text[:200]}")
            failures += 1
            continue
        body = resp.json()
        expected = case["expected_output"]
        got_types = [d["directive_type"] for d in sorted(body["directive_interpretation"], key=lambda x: x["note_index"])]
        want_types = [d["directive_type"] for d in sorted(expected["directive_interpretation"], key=lambda x: x["note_index"])]
        ok_types = got_types == want_types
        ok_hours_count = len(body["hourly_plan"]) == 24
        cost_delta = abs(body["total_cost_bdt"] - expected["total_cost_bdt"])
        status = "PASS" if ok_types and ok_hours_count else "CHECK"
        print(f"[{status}] {case['id']}: types={'match' if ok_types else f'got {got_types} want {want_types}'} "
              f"cost={body['total_cost_bdt']:.2f} (reference {expected['total_cost_bdt']:.2f}, delta {cost_delta:.2f})")
        if not (ok_types and ok_hours_count):
            failures += 1
    print(f"\n{len(data['cases']) - failures}/{len(data['cases'])} cases passed structural checks.")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the local server in one terminal**

Run: `cd F:\GridWise-LLM; .venv\Scripts\uvicorn app.main:app --reload`
Expected: server starts on `http://127.0.0.1:8000`, `/health` reachable.

- [ ] **Step 3: Set the real Gemini key and run the script against the live server**

Run (separate terminal): `cd F:\GridWise-LLM; $env:GEMINI_API_KEY="<your real key>"; .venv\Scripts\python scripts/run_public_samples.py`
Expected: all 10 cases print `[PASS]` or `[CHECK]` with a cost close to the reference; investigate any `[CHECK]` line before moving on — it means the LLM's real output diverged from the expected directive types, which is exactly what local testing is for.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_public_samples.py
git commit -m "test: add live regression script against all 10 public sample cases"
```

---

## Task 8: Dockerize

**Files:**
- Create: `F:\GridWise-LLM\Dockerfile`
- Create: `F:\GridWise-LLM\.dockerignore`

**Interfaces:**
- Produces: a pullable image exposing port 8000, binding `0.0.0.0`, reading `GEMINI_API_KEY` from the environment, no secrets baked in.

- [ ] **Step 1: Write Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends coinor-cbc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write .dockerignore**

```
.venv/
__pycache__/
*.pyc
.env
tests/
docs/
.git/
.pytest_cache/
```

- [ ] **Step 3: Build and run locally to verify**

Run: `cd F:\GridWise-LLM; docker build -t gridwise-llm:local .`
Expected: image builds successfully.

Run: `docker run -p 8000:8000 -e GEMINI_API_KEY="<your real key>" gridwise-llm:local`
Expected: container starts; in another terminal, `curl http://127.0.0.1:8000/health` returns `{"status":"ok"}`.

- [ ] **Step 4: Push to a registry**

Run: `docker tag gridwise-llm:local <your-dockerhub-username>/gridwise-llm:prelim` then `docker push <your-dockerhub-username>/gridwise-llm:prelim`
Expected: image pushed; note the exact tag for the README and submission form.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "chore: add Dockerfile with CBC solver, health check, non-root port binding"
```

---

## Task 9: Deploy to Render and verify externally

**Files:**
- Create: `F:\GridWise-LLM\render.yaml` (optional, but makes the Render setup one click / reproducible)

- [ ] **Step 1: Write render.yaml**

```yaml
services:
  - type: web
    name: gridwise-llm
    env: docker
    dockerfilePath: ./Dockerfile
    plan: free
    healthCheckPath: /health
    envVars:
      - key: GEMINI_API_KEY
        sync: false
```

- [ ] **Step 2: Push the repo to GitHub (private) and connect it in the Render dashboard**

Run: create a new **private** GitHub repository (per the rulebook, created after question reveal, kept private during the event), then:
`cd F:\GridWise-LLM; git remote add origin <your-private-repo-url>; git branch -M main; git push -u origin main`

In the Render dashboard: New → Web Service → connect the repo → Render detects `render.yaml` → set `GEMINI_API_KEY` in the environment variables UI (never commit it) → deploy.

- [ ] **Step 3: Verify externally**

Run (from your own machine, treating the Render URL as external): `curl https://<your-service>.onrender.com/health`
Expected: `{"status":"ok"}` within 60 seconds of a cold start.

Run: point `scripts/run_public_samples.py`'s `BASE_URL` at the Render URL temporarily (or pass it as an env var) and rerun it.
Expected: same pass results as the local run in Task 7.

- [ ] **Step 4: Commit**

```bash
git add render.yaml
git commit -m "chore: add Render deployment config"
```

---

## Task 10: README and final submission checklist pass

**Files:**
- Create: `F:\GridWise-LLM\README.md`

- [ ] **Step 1: Write the README**

```markdown
# GridWise LLM — BUP CSE Fest 2026 Preliminary

LLM-assisted 24-hour campus energy scheduler. Interprets 1-3 natural-language operator notes with Google
Gemini, deterministically guardrails that output, then solves an exact linear program for the cheapest valid
grid/solar/battery schedule.

## Architecture

Energy data + notes -> Gemini interpreter (`app/llm_interpreter.py`) -> deterministic guardrails
(`app/guardrails.py`, never trusts LLM output, falls back to `no_op` on anything malformed) -> LP optimizer
(`app/optimizer.py`, PuLP/CBC, exact global optimum) -> final replay self-check (`app/replay.py`) -> JSON
response. See the full flow in `app/main.py:optimize_energy`.

## Model / provider

Google Gemini, model `gemini-2.0-flash`, via the `google-generativeai` SDK, JSON response mode. Requires a
`GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey).

## Local quickstart (clean environment)

```bash
git clone <this-repo-url>
cd GridWise-LLM
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
copy .env.example .env        # then edit .env and paste in your real GEMINI_API_KEY
$env:GEMINI_API_KEY = (Get-Content .env | Select-String GEMINI_API_KEY).ToString().Split("=")[1]
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test it

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/optimize-energy -H "Content-Type: application/json" -d "@tests/fixtures/sample_request.json"

python scripts/run_public_samples.py
```

## Docker fallback

```bash
docker pull <your-dockerhub-username>/gridwise-llm:prelim
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key-here <your-dockerhub-username>/gridwise-llm:prelim
curl http://127.0.0.1:8000/health
```

## Dependencies

FastAPI, Uvicorn, Pydantic v2, `google-generativeai`, PuLP (CBC solver, installed via apt in the Docker image),
pytest, httpx.

## Known limitations

- Single LLM provider (Gemini); no automatic multi-provider failover — if Gemini is unreachable, the service
  falls back to marking all notes `no_op` and still returns a valid (but interpretation-degraded) schedule
  rather than failing the request.
- LP optimizer assumes all input scenarios are feasible, per the Problem Statement's guarantee that organizer
  scoring scenarios never require contradictory hard directives.

## Secret handling

No API keys or secrets are committed. `.env` is git-ignored; the Docker image reads `GEMINI_API_KEY` only from
the runtime environment.
```

- [ ] **Step 2: Create a real sample request fixture referenced by the README curl example**

Run: `cd F:\GridWise-LLM; python -c "import json; d=json.load(open('tests/fixtures/public_samples.json')); json.dump(d['cases'][0]['input'], open('tests/fixtures/sample_request.json','w'), indent=2)"`

- [ ] **Step 3: Walk the final pre-submit checklist from the Participant Guide §11 line by line**

Verify each of these manually against the deployed service and repo (all should already be true from Tasks 1-9; this step is the explicit gate before calling it done):
- `GET /health` reachable externally, returns `{"status":"ok"}`.
- `POST /optimize-energy` reachable externally, accepts 1-3 notes with the exact schema.
- Every note produces exactly one `directive_interpretation` entry in order; `no_op` semantics correct.
- `hourly_plan` obeys all directive + energy + battery rules (confirmed by `scripts/run_public_samples.py` and the `replay.py` self-check logging no violations).
- `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` match values recomputed from `hourly_plan` (guaranteed structurally since `main.py` computes them from the plan, never independently).
- README has a clean local quickstart, env var names, model/provider, run command, curl/sample tests, known limitations, no secrets.
- Repo created after question reveal, private during event, plan to flip to public after the deadline.
- Docker image pushed with an exact tag, documented pull/run commands verified to work, no baked-in secrets.
- Record the 3-minute video separately (not part of this plan's scope — screen-record a walkthrough of this architecture and a live demo call once the above is green).

- [ ] **Step 4: Commit**

```bash
git add README.md tests/fixtures/sample_request.json
git commit -m "docs: add README with architecture, quickstart, Docker fallback, and limitations"
```
