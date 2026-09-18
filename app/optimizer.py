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
