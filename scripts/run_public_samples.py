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
