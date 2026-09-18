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
