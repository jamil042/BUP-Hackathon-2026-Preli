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
