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
