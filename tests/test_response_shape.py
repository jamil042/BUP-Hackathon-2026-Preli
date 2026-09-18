import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
FIXTURES = json.loads(
    Path(__file__).parent.joinpath("fixtures/public_samples.json").read_text()
)

EXPECTED_KEYS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}


def _mock_interpret(expected):
    return [
        {
            "note_index": e["note_index"],
            "directive_type": e["directive_type"],
            "structured_adjustment": e["structured_adjustment"],
            "explanation": e["explanation"],
        }
        for e in expected
    ]


def test_structured_adjustment_shape_matches_spec_for_all_samples():
    for case in FIXTURES["cases"]:
        expected = case["expected_output"]["directive_interpretation"]
        with patch("app.main.interpret_notes", return_value=_mock_interpret(expected)):
            resp = client.post("/optimize-energy", json=case["input"])
        assert resp.status_code == 200, f"{case['id']}: {resp.status_code}"
        body = resp.json()
        for d in body["directive_interpretation"]:
            adj = d["structured_adjustment"]
            if d["directive_type"] == "no_op":
                assert adj is None, f"{case['id']}: no_op adjustment must be null, got {adj}"
                continue
            assert set(adj.keys()) == EXPECTED_KEYS[d["directive_type"]], (
                f"{case['id']} {d['directive_type']}: unexpected keys {sorted(adj)}"
            )
            assert adj["hours"] == sorted(adj["hours"]) and len(adj["hours"]) == len(
                set(adj["hours"])
            ), f"{case['id']}: hours not unique+sorted"



def test_extra_null_model_fields_are_stripped():
    case = FIXTURES["cases"][0]
    expected = [
        {
            "note_index": 0,
            "directive_type": "solar_reduction",
            "structured_adjustment": {
                "hours": [12, 13],
                "factor": 0.25,
                "minimum_energy_kwh": None,
                "max_grid_kwh": None,
            },
            "explanation": "cleaning window",
        },
        {
            "note_index": 1,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "unrelated",
        },
    ]
    with patch("app.main.interpret_notes", return_value=expected):
        resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200
    adj = resp.json()["directive_interpretation"][0]["structured_adjustment"]
    assert set(adj.keys()) == {"hours", "factor"}, f"got {adj}"