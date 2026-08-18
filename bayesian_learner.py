"""
bayesian_learner.py - Rete Bayesiana per rischio guasto/RUL class.

La rete non sostituisce la Knowledge Base: usa anche gli output inferiti dalla
KB come evidenza probabilistica.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

import config
import data_loader
import logic_engine

logging.getLogger("pgmpy").setLevel(logging.ERROR)

MODEL_CACHE: Optional[BayesianNetwork] = None
STATE_NAMES_CACHE: dict[str, list[str]] = {}
TARGET_CACHE: str = config.BAYES_TARGET

EVIDENCE_COLUMNS = [
    "OperatingRegime",
    "SensorAnomalyLevel",
    "ThermalState",
    "PressureState",
    "TrendState",
    "KB_ThermalStress",
    "KB_PressureInstability",
    "KB_AdverseTrend",
    "KB_DegradationEvidence",
]

RISK_ORDER = ["low", "medium", "high", "critical"]
RUL_ORDER = ["healthy", "warning", "degraded", "critical"]


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _ordered_states(values: pd.Series, target: Optional[str] = None) -> list[str]:
    unique = [str(v) for v in values.dropna().unique()]
    if target == "FailureRisk":
        return [s for s in RISK_ORDER if s in unique] + sorted(set(unique) - set(RISK_ORDER))
    if target == "RULClass":
        return [s for s in RUL_ORDER if s in unique] + sorted(set(unique) - set(RUL_ORDER))
    return sorted(unique)


def prepare_bayes_dataframe(
    df: Optional[pd.DataFrame] = None,
    subset: str = config.DEFAULT_SUBSET,
    data_dir: Optional[str] = None,
    target: str = config.BAYES_TARGET,
    use_kb: bool = True,
) -> pd.DataFrame:
    """
    Converte il DataFrame C-MAPSS nel formato discreto richiesto da pgmpy.

    Target ammessi:
    - FailureRisk: low/medium/high/critical;
    - RULClass: healthy/warning/degraded/critical.
    """

    if df is None:
        df = data_loader.get_clean_data(subset=subset, data_dir=data_dir)

    if use_kb and "kb_degradation_evidence" not in df.columns:
        df = logic_engine.annotate_with_kb(df)

    if target not in {"FailureRisk", "RULClass"}:
        raise ValueError("target deve essere 'FailureRisk' oppure 'RULClass'")

    if target == "FailureRisk":
        if "failure_risk" not in df.columns:
            raise ValueError("failure_risk mancante: usa split='train' o calcola RUL.")
        target_values = df["failure_risk"].astype(str)
    else:
        if "rul_class" not in df.columns:
            raise ValueError("rul_class mancante: usa split='train' o calcola RUL.")
        target_values = df["rul_class"].astype(str)

    data = pd.DataFrame(
        {
            "OperatingRegime": df["operating_regime"].astype(str),
            "SensorAnomalyLevel": df["sensor_anomaly_level"].astype(str),
            "ThermalState": df["thermal_state"].astype(str),
            "PressureState": df["pressure_state"].astype(str),
            "TrendState": df["trend_state"].astype(str),
            "KB_ThermalStress": df["kb_thermal_stress"].apply(_yes_no),
            "KB_PressureInstability": df["kb_pressure_instability"].apply(_yes_no),
            "KB_AdverseTrend": df["kb_adverse_trend"].apply(_yes_no),
            "KB_DegradationEvidence": df["kb_degradation_evidence"].apply(_yes_no),
            target: target_values,
            "engine_id": df["engine_id"].astype(int).values,
            "record_id": df["record_id"].astype(str).values,
        }
    ).dropna()

    if data[target].nunique() < 2:
        raise ValueError("Il target contiene una sola classe: impossibile valutare il modello.")

    return data


def prepare_evidence_dataframe(
    df: pd.DataFrame,
    use_kb: bool = True,
) -> pd.DataFrame:
    """
    Prepara solo le evidenze, senza richiedere il target.

    Serve per predire su test set o su dati reali dove RUL/failure_risk non sono
    noti.
    """

    if use_kb and "kb_degradation_evidence" not in df.columns:
        df = logic_engine.annotate_with_kb(df)

    data = pd.DataFrame(
        {
            "OperatingRegime": df["operating_regime"].astype(str),
            "SensorAnomalyLevel": df["sensor_anomaly_level"].astype(str),
            "ThermalState": df["thermal_state"].astype(str),
            "PressureState": df["pressure_state"].astype(str),
            "TrendState": df["trend_state"].astype(str),
            "KB_ThermalStress": df["kb_thermal_stress"].apply(_yes_no),
            "KB_PressureInstability": df["kb_pressure_instability"].apply(_yes_no),
            "KB_AdverseTrend": df["kb_adverse_trend"].apply(_yes_no),
            "KB_DegradationEvidence": df["kb_degradation_evidence"].apply(_yes_no),
            "engine_id": df["engine_id"].astype(int).values,
            "record_id": df["record_id"].astype(str).values,
        }
    ).dropna()

    return data


def _build_model(target: str = config.BAYES_TARGET) -> BayesianNetwork:
    """
    Struttura causale/diagnostica della rete.
    """

    return BayesianNetwork(
        [
            ("OperatingRegime", "SensorAnomalyLevel"),
            ("SensorAnomalyLevel", "ThermalState"),
            ("SensorAnomalyLevel", "PressureState"),
            ("SensorAnomalyLevel", "TrendState"),
            ("ThermalState", "KB_ThermalStress"),
            ("PressureState", "KB_PressureInstability"),
            ("TrendState", "KB_AdverseTrend"),
            ("KB_ThermalStress", "KB_DegradationEvidence"),
            ("KB_PressureInstability", "KB_DegradationEvidence"),
            ("KB_AdverseTrend", "KB_DegradationEvidence"),
            ("KB_DegradationEvidence", target),
            ("SensorAnomalyLevel", target),
        ]
    )


def _state_names_for(data: pd.DataFrame, target: str) -> dict[str, list[str]]:
    cols = EVIDENCE_COLUMNS + [target]
    return {col: _ordered_states(data[col], target if col == target else None) for col in cols}


def train_model(
    df: Optional[pd.DataFrame] = None,
    subset: str = config.DEFAULT_SUBSET,
    data_dir: Optional[str] = None,
    target: str = config.BAYES_TARGET,
    use_bayesian_estimator: bool = True,
) -> BayesianNetwork:
    """Addestra e mette in cache la rete bayesiana."""

    global MODEL_CACHE, STATE_NAMES_CACHE, TARGET_CACHE

    data = prepare_bayes_dataframe(df=df, subset=subset, data_dir=data_dir, target=target, use_kb=True)
    train_data = data[EVIDENCE_COLUMNS + [target]].copy()
    state_names = _state_names_for(train_data, target)

    model = _build_model(target)

    if use_bayesian_estimator:
        try:
            model.fit(
                train_data,
                estimator=BayesianEstimator,
                prior_type="dirichlet",
                pseudo_counts=config.BAYES_PSEUDO_COUNTS,
                state_names=state_names,
            )
        except TypeError:
            # Compatibilità con vecchie versioni pgmpy.
            model.fit(
                train_data,
                estimator=BayesianEstimator,
                prior_type="dirichlet",
                pseudo_counts=config.BAYES_PSEUDO_COUNTS,
            )
    else:
        model.fit(train_data, estimator=MaximumLikelihoodEstimator)

    MODEL_CACHE = model
    STATE_NAMES_CACHE = state_names
    TARGET_CACHE = target
    return model


def _clean_evidence(model: BayesianNetwork, evidence: dict[str, Any]) -> dict[str, str]:
    """Rimuove evidenze non note al modello o stati non visti in training."""

    clean: dict[str, str] = {}
    for col, value in evidence.items():
        if col not in EVIDENCE_COLUMNS:
            continue
        value = str(value)
        cpd = model.get_cpds(col)
        if cpd is None:
            continue
        known_states = set(cpd.state_names.get(col, []))
        if value in known_states:
            clean[col] = value
    return clean


def _probabilities_from_query(result, target: str) -> dict[str, float]:
    states = [str(s) for s in result.state_names.get(target, [])]
    values = np.asarray(result.values, dtype=float).ravel()
    return {state: float(prob) for state, prob in zip(states, values)}


def predict_failure_risk(
    evidence: dict[str, Any],
    model: Optional[BayesianNetwork] = None,
    target: Optional[str] = None,
) -> dict[str, Any]:
    """
    Predice la distribuzione del target per una singola evidenza.

    Restituisce:
    {
        "prediction": classe più probabile,
        "probabilities": {classe: probabilità}
    }
    """

    global MODEL_CACHE

    if model is None:
        if MODEL_CACHE is None:
            train_model(target=target or TARGET_CACHE)
        model = MODEL_CACHE

    target = target or TARGET_CACHE
    infer = VariableElimination(model)
    evidence = _clean_evidence(model, evidence)

    if evidence:
        result = infer.query(variables=[target], evidence=evidence, show_progress=False)
    else:
        result = infer.query(variables=[target], show_progress=False)

    probs = _probabilities_from_query(result, target)
    prediction = max(probs, key=probs.get)
    return {"prediction": prediction, "probabilities": probs}


def _evidence_from_prepared_row(row: pd.Series) -> dict[str, str]:
    return {col: str(row[col]) for col in EVIDENCE_COLUMNS if col in row.index}


def predict_dataframe(
    df: pd.DataFrame,
    model: Optional[BayesianNetwork] = None,
    target: str = config.BAYES_TARGET,
) -> pd.DataFrame:
    """Aggiunge predizioni bayesiane a un DataFrame C-MAPSS."""

    if "kb_degradation_evidence" not in df.columns:
        df = logic_engine.annotate_with_kb(df)

    prepared = prepare_evidence_dataframe(df=df, use_kb=False)

    if model is None:
        model = MODEL_CACHE
        if model is None:
            model = train_model(df=df, target=target)

    predictions = []
    probs_rows = []
    for _, row in prepared.iterrows():
        result = predict_failure_risk(_evidence_from_prepared_row(row), model=model, target=target)
        predictions.append(result["prediction"])
        probs_rows.append(result["probabilities"])

    out = df.copy().reset_index(drop=True)
    out[f"predicted_{target}"] = predictions

    all_states = sorted({state for probs in probs_rows for state in probs})
    for state in all_states:
        out[f"prob_{target}_{state}"] = [probs.get(state, 0.0) for probs in probs_rows]

    return out


def _multiclass_brier(y_true: list[str], prob_rows: list[dict[str, float]], labels: list[str]) -> float:
    total = 0.0
    for true, probs in zip(y_true, prob_rows):
        total += sum((probs.get(label, 0.0) - (1.0 if label == true else 0.0)) ** 2 for label in labels)
    return float(total / max(len(y_true), 1))


def _fit_fold_model(train_data: pd.DataFrame, target: str, state_names: dict[str, list[str]]) -> BayesianNetwork:
    model = _build_model(target)
    fit_data = train_data[EVIDENCE_COLUMNS + [target]].copy()
    try:
        model.fit(
            fit_data,
            estimator=BayesianEstimator,
            prior_type="dirichlet",
            pseudo_counts=config.BAYES_PSEUDO_COUNTS,
            state_names=state_names,
        )
    except TypeError:
        model.fit(
            fit_data,
            estimator=BayesianEstimator,
            prior_type="dirichlet",
            pseudo_counts=config.BAYES_PSEUDO_COUNTS,
        )
    return model


def cross_validate_bayesian_network(
    df: Optional[pd.DataFrame] = None,
    subset: str = config.DEFAULT_SUBSET,
    data_dir: Optional[str] = None,
    k: int = 5,
    target: str = config.BAYES_TARGET,
    random_state: int = 42,  # mantenuto per coerenza API; GroupKFold non lo usa.
) -> dict[str, float]:
    """
    Valuta la rete bayesiana con GroupKFold per evitare leakage tra cicli dello stesso motore.
    """

    del random_state

    data = prepare_bayes_dataframe(df=df, subset=subset, data_dir=data_dir, target=target, use_kb=True)
    groups = data["engine_id"].values
    n_groups = len(np.unique(groups))
    if n_groups < k:
        raise ValueError(f"Impossibile usare k={k}: motori disponibili={n_groups}")

    labels = _ordered_states(data[target], target)
    state_names = _state_names_for(data[EVIDENCE_COLUMNS + [target]], target)

    cv = GroupKFold(n_splits=k)

    accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    f1_macros: list[float] = []
    briers: list[float] = []

    for train_idx, test_idx in cv.split(data, data[target], groups):
        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()

        model = _fit_fold_model(train_data, target, state_names)

        y_true: list[str] = []
        y_pred: list[str] = []
        prob_rows: list[dict[str, float]] = []

        for _, row in test_data.iterrows():
            result = predict_failure_risk(_evidence_from_prepared_row(row), model=model, target=target)
            y_true.append(str(row[target]))
            y_pred.append(str(result["prediction"]))
            prob_rows.append(result["probabilities"])

        accuracies.append(accuracy_score(y_true, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_true, y_pred))
        f1_macros.append(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
        briers.append(_multiclass_brier(y_true, prob_rows, labels))

    results = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "balanced_accuracy_mean": float(np.mean(balanced_accuracies)),
        "balanced_accuracy_std": float(np.std(balanced_accuracies, ddof=1)),
        "f1_macro_mean": float(np.mean(f1_macros)),
        "f1_macro_std": float(np.std(f1_macros, ddof=1)),
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers, ddof=1)),
    }

    print("\nRisultati Rete Bayesiana - GroupKFold CV")
    print("------------------------------------------")
    print(f"Accuracy:          {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy_mean']:.3f} ± {results['balanced_accuracy_std']:.3f}")
    print(f"F1-Macro:          {results['f1_macro_mean']:.3f} ± {results['f1_macro_std']:.3f}")
    print(f"Brier Score:       {results['brier_mean']:.3f} ± {results['brier_std']:.3f}")

    return results


if __name__ == "__main__":
    cross_validate_bayesian_network(k=5)
