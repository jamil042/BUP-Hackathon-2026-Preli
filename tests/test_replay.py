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
