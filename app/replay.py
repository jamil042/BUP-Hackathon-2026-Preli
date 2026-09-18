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
