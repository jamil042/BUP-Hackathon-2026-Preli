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
