"""
logic_engine.py - Knowledge Base diagnostica in pyDatalog.

La KB è il cuore simbolico del progetto: non memorizza soltanto righe del
Dataset, ma inferisce stati diagnostici a partire da fatti sensoriali
simbolici.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pyDatalog import pyDatalog

import data_loader


pyDatalog.create_terms(
    "engine, thermal_state_fact, pressure_state_fact, rotation_state_fact, "
    "trend_state_fact, anomaly_level_fact, operating_regime_fact, "
    "thermal_stress, thermal_critical, pressure_instability, rotation_instability, "
    "adverse_trend, multiple_sensor_anomaly, degradation_evidence, "
    "critical_engine, urgent_maintenance, needs_inspection, needs_repair, "
    "needs_replacement, E"
)

_LAST_ID_COLUMN = "record_id"


def _normalize_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "si", "sì"}
    return bool(value)


def setup_logic(
    df: Optional[pd.DataFrame] = None,
    subset: str = "FD001",
    data_dir: Optional[str] = None,
    latest_only: bool = False,
    id_column: str = "record_id",
) -> None:
    """
    Inizializza fatti e regole della KB.

    Nota importante: le regole NON usano RUL o target reali. In questo modo la
    KB può essere valutata contro il target senza leakage.
    """

    global _LAST_ID_COLUMN
    _LAST_ID_COLUMN = id_column

    pyDatalog.clear()

    if df is None:
        df = data_loader.get_clean_data(subset=subset, data_dir=data_dir)

    if latest_only:
        df = data_loader.get_latest_engine_state(df)

    if id_column not in df.columns:
        raise ValueError(f"Colonna id mancante per la KB: {id_column}")

    required = {
        "thermal_state",
        "pressure_state",
        "rotation_state",
        "trend_state",
        "sensor_anomaly_level",
        "operating_regime",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per la KB: {missing}")

    for _, row in df.iterrows():
        rid = str(row[id_column])
        pyDatalog.assert_fact("engine", rid)
        pyDatalog.assert_fact("thermal_state_fact", rid, str(row["thermal_state"]))
        pyDatalog.assert_fact("pressure_state_fact", rid, str(row["pressure_state"]))
        pyDatalog.assert_fact("rotation_state_fact", rid, str(row["rotation_state"]))
        pyDatalog.assert_fact("trend_state_fact", rid, str(row["trend_state"]))
        pyDatalog.assert_fact("anomaly_level_fact", rid, str(row["sensor_anomaly_level"]))
        pyDatalog.assert_fact("operating_regime_fact", rid, str(row["operating_regime"]))

    # ------------------------------------------------------------------
    # Regole diagnostiche.
    # ------------------------------------------------------------------

    # Stress termico: più di una semplice soglia, perché si basa su stati
    # simbolici derivati da gruppi di sensori.
    thermal_stress(E) <= thermal_state_fact(E, "moderate")
    thermal_stress(E) <= thermal_state_fact(E, "severe")
    thermal_critical(E) <= thermal_state_fact(E, "severe")

    # Instabilità di pressione e rotazione.
    pressure_instability(E) <= pressure_state_fact(E, "moderate")
    pressure_instability(E) <= pressure_state_fact(E, "severe")
    rotation_instability(E) <= rotation_state_fact(E, "moderate")
    rotation_instability(E) <= rotation_state_fact(E, "severe")

    # Trend avverso: diversi sensori mostrano peggioramento coerente.
    adverse_trend(E) <= trend_state_fact(E, "moderate")
    adverse_trend(E) <= trend_state_fact(E, "severe")

    # Anomalia multisensore.
    multiple_sensor_anomaly(E) <= anomaly_level_fact(E, "moderate")
    multiple_sensor_anomaly(E) <= anomaly_level_fact(E, "severe")

    # Evidenza di degrado: regole combinatorie, non singola soglia.
    degradation_evidence(E) <= thermal_stress(E) & adverse_trend(E)
    degradation_evidence(E) <= pressure_instability(E) & multiple_sensor_anomaly(E)
    degradation_evidence(E) <= thermal_stress(E) & pressure_instability(E)
    degradation_evidence(E) <= rotation_instability(E) & adverse_trend(E)

    # Stato critico inferito soltanto da sintomi, non dal target RUL.
    critical_engine(E) <= thermal_critical(E) & pressure_instability(E)
    critical_engine(E) <= degradation_evidence(E) & anomaly_level_fact(E, "severe")
    critical_engine(E) <= pressure_instability(E) & adverse_trend(E) & multiple_sensor_anomaly(E)

    # Decisione manutentiva simbolica.
    urgent_maintenance(E) <= critical_engine(E)
    urgent_maintenance(E) <= degradation_evidence(E) & adverse_trend(E)

    needs_inspection(E) <= degradation_evidence(E)
    needs_repair(E) <= urgent_maintenance(E) & pressure_instability(E)
    needs_replacement(E) <= critical_engine(E)


def _query_set(predicate) -> set[str]:
    result = predicate(E)
    if not result:
        return set()
    return {str(row[0]) for row in result}


def query_degradation_evidence() -> set[str]:
    return _query_set(degradation_evidence)


def query_critical_engines() -> set[str]:
    return _query_set(critical_engine)


def query_urgent_maintenance() -> set[str]:
    return _query_set(urgent_maintenance)


def query_needs_inspection() -> set[str]:
    return _query_set(needs_inspection)


def query_needs_repair() -> set[str]:
    return _query_set(needs_repair)


def query_needs_replacement() -> set[str]:
    return _query_set(needs_replacement)


def get_deadline_days(record_id: str) -> Optional[int]:
    """Deadline simbolica derivata dalle inferenze KB."""

    rid = str(record_id)
    if rid in query_critical_engines():
        return 2
    if rid in query_urgent_maintenance():
        return 4
    if rid in query_degradation_evidence():
        return 7
    return None

def annotate_with_kb(
    df: pd.DataFrame,
    id_column: str = "record_id",
) -> pd.DataFrame:
    """
    Versione veloce dell'annotazione KB.

    Implementa in pandas le stesse regole definite nella Knowledge Base pyDatalog.
    La KB resta comunque disponibile tramite setup_logic() e query_*(), ma per
    annotare migliaia di record è molto più efficiente usare operazioni vettoriali.
    """

    out = df.copy()

    required = {
        "thermal_state",
        "pressure_state",
        "rotation_state",
        "trend_state",
        "sensor_anomaly_level",
    }

    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per annotazione KB: {missing}")

    # ------------------------------------------------------------
    # Regole diagnostiche intermedie
    # ------------------------------------------------------------

    out["kb_thermal_stress"] = out["thermal_state"].astype(str).isin(
        ["moderate", "severe"]
    )

    thermal_critical = out["thermal_state"].astype(str).eq("severe")

    out["kb_pressure_instability"] = out["pressure_state"].astype(str).isin(
        ["moderate", "severe"] 
    )

    rotation_instability = out["rotation_state"].astype(str).isin(
        ["moderate", "severe"]
    )

    out["kb_adverse_trend"] = out["trend_state"].astype(str).isin(
        ["moderate", "severe"]
    )

    multiple_sensor_anomaly = out["sensor_anomaly_level"].astype(str).isin(
        ["moderate", "severe"]
    )

    anomaly_severe = out["sensor_anomaly_level"].astype(str).eq("severe")

    # ------------------------------------------------------------
    # Regole combinatorie di degrado
    # ------------------------------------------------------------

    out["kb_degradation_evidence"] = (
        (out["kb_thermal_stress"] & out["kb_adverse_trend"])
        | (out["kb_pressure_instability"] & multiple_sensor_anomaly)
        | (out["kb_thermal_stress"] & out["kb_pressure_instability"])
        | (rotation_instability & out["kb_adverse_trend"])
    )

    # ------------------------------------------------------------
    # Regole di criticità
    # ------------------------------------------------------------

    out["kb_critical_engine"] = (
        (thermal_critical & out["kb_pressure_instability"])
        | (out["kb_degradation_evidence"] & anomaly_severe)
        | (
            out["kb_pressure_instability"]
            & out["kb_adverse_trend"]
            & multiple_sensor_anomaly
        )
    )

    # ------------------------------------------------------------
    # Decisione manutentiva simbolica
    # ------------------------------------------------------------

    out["kb_urgent_maintenance"] = (
        out["kb_critical_engine"]
        | (out["kb_degradation_evidence"] & out["kb_adverse_trend"])
    )

    out["kb_needs_inspection"] = out["kb_degradation_evidence"]

    out["kb_needs_repair"] = (
        out["kb_urgent_maintenance"] & out["kb_pressure_instability"]
    )

    out["kb_needs_replacement"] = out["kb_critical_engine"]

    # ------------------------------------------------------------
    # Azione raccomandata
    # Priorità: replacement > repair > inspection > none
    # ------------------------------------------------------------

    out["kb_recommended_action"] = "none"

    out.loc[out["kb_needs_inspection"], "kb_recommended_action"] = "inspection"
    out.loc[out["kb_needs_repair"], "kb_recommended_action"] = "repair"
    out.loc[out["kb_needs_replacement"], "kb_recommended_action"] = "replacement"

    # ------------------------------------------------------------
    # Deadline simbolica
    # Priorità: critical = 2 giorni, urgent = 4, degradation = 7
    # ------------------------------------------------------------

    out["kb_deadline_days"] = pd.NA

    out.loc[out["kb_degradation_evidence"], "kb_deadline_days"] = 7
    out.loc[out["kb_urgent_maintenance"], "kb_deadline_days"] = 4
    out.loc[out["kb_critical_engine"], "kb_deadline_days"] = 2

    return out

if __name__ == "__main__":
    data = data_loader.get_clean_data()
    annotated = annotate_with_kb(data)
    print(annotated[["record_id", "kb_degradation_evidence", "kb_urgent_maintenance", "kb_recommended_action"]].head())
