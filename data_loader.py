"""
data_loader.py - Loader e preprocessing per NASA C-MAPSS.

Sostituisce il vecchio dominio degli esami con un dominio di manutenzione
predittiva dei motori turbofan.

Il loader produce un DataFrame già utilizzabile da:
- Knowledge Base pyDatalog;
- Rete Bayesiana pgmpy;
- CSP di pianificazione manutentiva.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config


ID_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_data_dir(data_dir: Optional[str | os.PathLike] = None) -> Path:
    """
    Trova la cartella che contiene i file C-MAPSS.

    Ordine di ricerca:
    1. parametro esplicito data_dir;
    2. variabile d'ambiente CMAPSS_DATA_DIR;
    3. data/CMAPSSData, data/CMAPSS, CMAPSSData, CMAPSS accanto ai .py.
    """

    candidates: list[Path] = []

    if data_dir is not None:
        candidates.append(Path(data_dir))

    env_dir = os.getenv("CMAPSS_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    root = _project_root()
    candidates.extend(root / candidate for candidate in config.DATA_DIR_CANDIDATES)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    tried = "\n - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Cartella C-MAPSS non trovata. Metti i file in data/CMAPSSData "
        "oppure passa --data-dir. Percorsi provati:\n - " + tried
    )


def _subset_name(subset: str) -> str:
    subset = str(subset).upper().replace("TRAIN_", "").replace("TEST_", "")
    if not subset.startswith("FD"):
        subset = f"FD{subset}"
    return subset


def load_cmapss_subset(
    subset: str = config.DEFAULT_SUBSET,
    split: str = "train",
    data_dir: Optional[str | os.PathLike] = None,
) -> pd.DataFrame:
    """
    Carica un file C-MAPSS, ad esempio train_FD001.txt.

    Il file non ha header e contiene:
    engine_id, cycle, 3 operational settings, 21 sensori.
    """

    subset = _subset_name(subset)
    split = split.lower().strip()
    if split not in {"train", "test"}:
        raise ValueError("split deve essere 'train' oppure 'test'")

    base_dir = resolve_data_dir(data_dir)
    path = base_dir / f"{split}_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    df = df.dropna(axis=1, how="all")

    # Alcuni parser possono creare colonne extra se ci sono spazi finali anomali.
    if len(df.columns) > len(CMAPSS_COLUMNS):
        df = df.iloc[:, : len(CMAPSS_COLUMNS)]
        df.columns = CMAPSS_COLUMNS

    df["subset"] = subset
    df["split"] = split
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    numeric_cols = SETTING_COLUMNS + SENSOR_COLUMNS
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # Identificatore univoco di record: utile per KB e debug.
    df["record_id"] = (
        df["subset"].astype(str)
        + "_E"
        + df["engine_id"].astype(str)
        + "_C"
        + df["cycle"].astype(str)
    )

    return df


def add_train_rul(df: pd.DataFrame, max_rul_cap: int = config.MAX_RUL_CAP) -> pd.DataFrame:
    """
    Calcola RUL sui dati di training.

    Per ogni motore, l'ultimo ciclo osservato è considerato ciclo di guasto.
    RUL = max_cycle_engine - cycle.
    """

    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["max_cycle"] = max_cycle.astype(int)
    df["rul"] = (max_cycle - df["cycle"]).astype(int)
    df["rul_capped"] = df["rul"].clip(upper=max_rul_cap).astype(int)
    df["rul_class"] = df["rul"].apply(classify_rul)
    df["failure_risk"] = df["rul_class"].map(
        {
            "healthy": "low",
            "warning": "medium",
            "degraded": "high",
            "critical": "critical",
        }
    )
    df["urgent_label"] = (df["rul"] <= config.RUL_DEGRADED_THRESHOLD).astype(int)
    return df


def classify_rul(rul: float | int) -> str:
    """Restituisce una classe simbolica di Remaining Useful Life."""

    rul = float(rul)
    if rul <= config.RUL_CRITICAL_THRESHOLD:
        return "critical"
    if rul <= config.RUL_DEGRADED_THRESHOLD:
        return "degraded"
    if rul <= config.RUL_WARNING_THRESHOLD:
        return "warning"
    return "healthy"


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0))
    if np.isnan(value) or value < 1e-9:
        return 1.0
    return value


def add_window_features(
    df: pd.DataFrame,
    window: int = config.WINDOW_SIZE,
    baseline_window: int = config.BASELINE_WINDOW,
) -> pd.DataFrame:
    """
    Crea feature rolling e anomalie normalizzate rispetto alla baseline iniziale.

    Non usa il target RUL: questo evita leakage tra label e feature diagnostiche.
    """

    df = df.sort_values(["engine_id", "cycle"]).copy()

    grouped = df.groupby("engine_id", group_keys=False)

    for sensor in SENSOR_COLUMNS:
        roll = grouped[sensor].rolling(window=window, min_periods=1)
        df[f"{sensor}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{sensor}_roll_std"] = roll.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
        df[f"{sensor}_trend"] = grouped[sensor].transform(
            lambda s: (s - s.shift(window - 1)) / max(window - 1, 1)
        ).fillna(0.0)

        baseline_mean = grouped[sensor].transform(lambda s: float(s.head(baseline_window).mean()))
        baseline_std = grouped[sensor].transform(lambda s: _safe_std(s.head(baseline_window)))
        df[f"{sensor}_z"] = (df[f"{sensor}_roll_mean"] - baseline_mean) / baseline_std

    return df


def _count_abs_z(df: pd.DataFrame, sensors: Iterable[str], threshold: float) -> pd.Series:
    cols = [f"{s}_z" for s in sensors]
    return (df[cols].abs() >= threshold).sum(axis=1)


def _count_negative_trends(df: pd.DataFrame, sensors: Iterable[str], eps: float) -> pd.Series:
    cols = [f"{s}_trend" for s in sensors]
    return (df[cols] <= -abs(eps)).sum(axis=1)


def _state_from_count(count: int, moderate_at: int, severe_at: int) -> str:
    if count >= severe_at:
        return "severe"
    if count >= moderate_at:
        return "moderate"
    if count > 0:
        return "mild"
    return "normal"


def add_symbolic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma feature numeriche in stati simbolici utili per KB e Bayes.
    """

    df = df.copy()

    df["thermal_anomaly_count"] = _count_abs_z(df, config.THERMAL_SENSORS, config.Z_ANOMALY)
    df["pressure_anomaly_count"] = _count_abs_z(df, config.PRESSURE_SENSORS, config.Z_ANOMALY)
    df["rotation_anomaly_count"] = _count_abs_z(df, config.ROTATION_SENSORS, config.Z_ANOMALY)
    df["critical_sensor_count"] = _count_abs_z(df, SENSOR_COLUMNS, config.Z_CRITICAL)
    df["negative_trend_count"] = _count_negative_trends(df, SENSOR_COLUMNS, config.TREND_EPS)

    df["thermal_state"] = df["thermal_anomaly_count"].apply(lambda x: _state_from_count(int(x), 2, 4))
    df["pressure_state"] = df["pressure_anomaly_count"].apply(lambda x: _state_from_count(int(x), 2, 3))
    df["rotation_state"] = df["rotation_anomaly_count"].apply(lambda x: _state_from_count(int(x), 1, 2))
    df["trend_state"] = df["negative_trend_count"].apply(lambda x: _state_from_count(int(x), 5, 9))

    total_anomalies = (
        df["thermal_anomaly_count"]
        + df["pressure_anomaly_count"]
        + df["rotation_anomaly_count"]
        + df["critical_sensor_count"]
    )
    df["sensor_anomaly_level"] = total_anomalies.apply(lambda x: _state_from_count(int(x), 4, 8))

    # Regime operativo simbolico basato sui tre operational settings.
    # Usiamo quantili robusti e pochi stati per evitare CPD troppo sparse.
    op1 = df["op_setting_1"]
    q1, q2 = op1.quantile([0.33, 0.66])
    df["operating_regime"] = np.select(
        [op1 <= q1, op1 <= q2],
        ["low", "medium"],
        default="high",
    )

    return df


def get_clean_data(
    subset: str = config.DEFAULT_SUBSET,
    split: str = "train",
    data_dir: Optional[str | os.PathLike] = None,
    window: int = config.WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Entry point usato dagli altri moduli.

    Per split='train' aggiunge anche RUL e target sperimentali.
    """

    df = load_cmapss_subset(subset=subset, split=split, data_dir=data_dir)

    if split.lower() == "train":
        df = add_train_rul(df)

    df = add_window_features(df, window=window)
    df = add_symbolic_features(df)

    return df


def get_latest_engine_state(df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
    """
    Restituisce l'ultima osservazione disponibile per ogni motore.

    È il formato naturale per il CSP, che deve pianificare interventi sui motori
    nello stato più recente conosciuto.
    """

    if df is None:
        df = get_clean_data(**kwargs)

    idx = df.groupby("engine_id")["cycle"].idxmax()
    latest = df.loc[idx].sort_values("engine_id").reset_index(drop=True)
    return latest


if __name__ == "__main__":
    data = get_clean_data()
    print(data.shape)
    print(data[["engine_id", "cycle", "rul", "rul_class", "failure_risk", "sensor_anomaly_level"]].head())
