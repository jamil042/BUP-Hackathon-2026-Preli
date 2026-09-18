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
        raw_directives = interpret_notes(request.operator_notes, request.battery.capacity_kwh)
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
