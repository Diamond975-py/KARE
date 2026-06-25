"""
maintenance_optimizer.py - CSP per pianificazione manutentiva.

Il CSP assegna motori a slot, tecnici e tipi di intervento rispettando vincoli
operativi: deadline, competenze tecniche, budget giornaliero e capacità.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from constraint import Problem

import bayesian_learner
import config
import data_loader
import logic_engine


RISK_WEIGHT = {
    "low": 1,
    "medium": 2,
    "high": 4,
    "critical": 6,
}


def _risk_from_row(row: pd.Series) -> str:
    for col in ["predicted_FailureRisk", "failure_risk"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "medium" if bool(row.get("kb_degradation_evidence", False)) else "low"


def _deadline_from_row(row: pd.Series, max_days: int) -> int:
    if "kb_deadline_days" in row and pd.notna(row["kb_deadline_days"]):
        return min(int(row["kb_deadline_days"]), max_days)

    risk = _risk_from_row(row)
    if risk == "critical":
        return min(2, max_days)
    if risk == "high":
        return min(4, max_days)
    return max_days


def _action_from_row(row: pd.Series) -> str:
    risk = _risk_from_row(row)

    if bool(row.get("kb_needs_replacement", False)) or risk == "critical":
        return "replacement"
    if bool(row.get("kb_needs_repair", False)) or risk == "high":
        return "repair"
    return "inspection"


def _priority(row: pd.Series) -> float:
    risk = _risk_from_row(row)
    score = RISK_WEIGHT.get(risk, 1)
    if bool(row.get("kb_urgent_maintenance", False)):
        score += 3
    if bool(row.get("kb_critical_engine", False)):
        score += 3
    if bool(row.get("kb_degradation_evidence", False)):
        score += 1
    return float(score)


def _prepare_engine_states(
    df: Optional[pd.DataFrame] = None,
    subset: str = config.DEFAULT_SUBSET,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Carica ultima osservazione per motore e aggiunge KB + predizioni Bayesiane."""

    if df is None:
        full = data_loader.get_clean_data(subset=subset, data_dir=data_dir)
        bayesian_learner.train_model(df=full)
        latest = data_loader.get_latest_engine_state(full)
        latest = logic_engine.annotate_with_kb(latest)
        latest = bayesian_learner.predict_dataframe(latest)
        return latest

    out = df.copy()
    if "kb_degradation_evidence" not in out.columns:
        out = logic_engine.annotate_with_kb(out)
    if "predicted_FailureRisk" not in out.columns and "failure_risk" in out.columns:
        try:
            out = bayesian_learner.predict_dataframe(out)
        except Exception:
            # Il CSP può comunque funzionare usando KB e failure_risk se disponibile.
            pass
    return out


def _candidate_table(
    engine_states: pd.DataFrame,
    max_days: int,
    max_candidates: int,
) -> pd.DataFrame:
    df = engine_states.copy()
    df["maintenance_risk"] = df.apply(_risk_from_row, axis=1)
    df["maintenance_action"] = df.apply(_action_from_row, axis=1)
    df["maintenance_deadline"] = df.apply(lambda row: _deadline_from_row(row, max_days), axis=1)
    df["maintenance_priority"] = df.apply(_priority, axis=1)

    # Il CSP pianifica solo i casi da trattare: rischio medio/alto/critico o KB positiva.
    mask = (
        df["maintenance_risk"].isin(["medium", "high", "critical"])
        | df.get("kb_degradation_evidence", False).astype(bool)
        | df.get("kb_urgent_maintenance", False).astype(bool)
    )

    candidates = df[mask].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["maintenance_priority", "maintenance_deadline", "engine_id"],
        ascending=[False, True, True],
    ).head(max_candidates)

    return candidates.reset_index(drop=True)


def _domain_for_engine(row: pd.Series, max_days: int, slots: list[str], technicians: dict[str, set[str]]):
    deadline = int(row["maintenance_deadline"])
    action = str(row["maintenance_action"])
    domain = []
    for day in range(1, min(deadline, max_days) + 1):
        for slot in slots:
            for tech, skills in technicians.items():
                if action in skills:
                    domain.append((day, slot, tech, action))
    return domain


def _solution_to_plan(solution: dict, candidates: pd.DataFrame, runtime: float) -> dict:
    rows = []
    total_cost = 0
    weighted_earliness = 0.0
    deadline_satisfied = 0
    critical_scheduled = 0

    candidate_by_id = {str(row["engine_id"]): row for _, row in candidates.iterrows()}

    for engine_key, assignment in solution.items():
        day, slot, technician, action = assignment
        row = candidate_by_id[str(engine_key)]
        risk = str(row["maintenance_risk"])
        deadline = int(row["maintenance_deadline"])
        cost = config.CSP_ACTION_COSTS[action]
        total_cost += cost

        if day <= deadline:
            deadline_satisfied += 1
        if risk == "critical" or bool(row.get("kb_critical_engine", False)):
            critical_scheduled += 1

        weighted_earliness += RISK_WEIGHT.get(risk, 1) * (deadline - day + 1)

        rows.append(
            {
                "engine_id": int(row["engine_id"]),
                "record_id": str(row.get("record_id", engine_key)),
                "risk": risk,
                "kb_urgent": bool(row.get("kb_urgent_maintenance", False)),
                "deadline_day": deadline,
                "scheduled_day": int(day),
                "slot": slot,
                "technician": technician,
                "action": action,
                "cost": cost,
            }
        )

    n = max(len(rows), 1)
    score = weighted_earliness - (total_cost / 10000.0)

    return {
        "schedule": sorted(rows, key=lambda x: (x["scheduled_day"], x["slot"], x["technician"])),
        "total_cost": int(total_cost),
        "deadline_satisfaction_rate": float(deadline_satisfied / n),
        "critical_engines_scheduled": int(critical_scheduled),
        "scheduled_engines": int(len(rows)),
        "score": float(score),
        "runtime_seconds": float(runtime),
    }


def find_maintenance_schedule(
    engine_states: Optional[pd.DataFrame] = None,
    subset: str = config.DEFAULT_SUBSET,
    data_dir: Optional[str] = None,
    max_days: int = config.CSP_DAYS,
    slots: Optional[list[str]] = None,
    technicians: Optional[dict[str, set[str]]] = None,
    daily_budget: int = config.CSP_DAILY_BUDGET,
    max_engines_per_day: int = config.CSP_MAX_ENGINES_PER_DAY,
    max_candidates: int = config.CSP_MAX_CANDIDATES,
    max_solutions: int = config.CSP_MAX_SOLUTIONS,
) -> list[dict]:
    """
    Trova uno o più piani manutentivi validi tramite CSP.
    """

    start = time.perf_counter()
    slots = slots or config.CSP_SLOTS
    technicians = technicians or config.CSP_TECHNICIANS

    engine_states = _prepare_engine_states(engine_states, subset=subset, data_dir=data_dir)
    candidates = _candidate_table(engine_states, max_days=max_days, max_candidates=max_candidates)

    if candidates.empty:
        return []

    problem = Problem()
    engine_keys = [str(e) for e in candidates["engine_id"].tolist()]
    candidate_by_key = {str(row["engine_id"]): row for _, row in candidates.iterrows()}

    for key in engine_keys:
        domain = _domain_for_engine(candidate_by_key[key], max_days, slots, technicians)
        if not domain:
            return []
        problem.addVariable(key, domain)

    def no_technician_conflict(*assignments):
        occupied = set()
        for day, slot, tech, _action in assignments:
            key = (day, slot, tech)
            if key in occupied:
                return False
            occupied.add(key)
        return True

    def daily_budget_constraint(*assignments):
        cost_by_day = {}
        for day, _slot, _tech, action in assignments:
            cost_by_day[day] = cost_by_day.get(day, 0) + config.CSP_ACTION_COSTS[action]
        return all(cost <= daily_budget for cost in cost_by_day.values())

    def daily_capacity_constraint(*assignments):
        count_by_day = {}
        for day, _slot, _tech, _action in assignments:
            count_by_day[day] = count_by_day.get(day, 0) + 1
        return all(count <= max_engines_per_day for count in count_by_day.values())

    problem.addConstraint(no_technician_conflict, engine_keys)
    problem.addConstraint(daily_budget_constraint, engine_keys)
    problem.addConstraint(daily_capacity_constraint, engine_keys)

    plans: list[dict] = []
    for i, solution in enumerate(problem.getSolutionIter()):
        runtime = time.perf_counter() - start
        plans.append(_solution_to_plan(solution, candidates, runtime))
        if i + 1 >= max_solutions:
            break

    plans.sort(key=lambda p: p["score"], reverse=True)
    return plans


if __name__ == "__main__":
    schedules = find_maintenance_schedule()
    if not schedules:
        print("Nessun piano manutentivo trovato.")
    else:
        best = schedules[0]
        print(f"Score: {best['score']:.3f}")
        print(f"Costo totale: {best['total_cost']}")
        for item in best["schedule"]:
            print(item)
