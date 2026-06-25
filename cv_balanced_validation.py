"""
cv_balanced_validation.py - Baseline supervisionate per KARE.

Valuta modelli supervisionati standard come confronto esterno rispetto al sistema
KARE KB + Bayes + CSP.

Nota:
- Questo script NON valuta la rete bayesiana pgmpy.
- La rete bayesiana va valutata separatamente con cv_bayes.py.
- Qui evitiamo import di pgmpy per non avere problemi con Python 3.9.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import config
import data_loader
import logic_engine


def _one_hot_encoder_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _format_mean_std(values):
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def _make_pipeline(model, feature_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", _one_hot_encoder_dense(), feature_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", model),
    ])


def run_model_comparison(
    subset=config.DEFAULT_SUBSET,
    data_dir=None,
    k=5,
    max_rows=None,
    skip_kb=False,
):
    total_start = time.perf_counter()

    print("=" * 90)
    print("KARE - BASELINE SUPERVISIONATE CON GROUPKFOLD")
    print("=" * 90)
    print(f"Subset: {subset}")
    print(f"k fold: {k}")
    print(f"data_dir: {data_dir if data_dir else 'default da config/data_loader'}")
    print(f"max_rows: {max_rows if max_rows else 'nessun limite'}")
    print(f"skip_kb: {skip_kb}")
    print("-" * 90, flush=True)

    print("[1/6] Caricamento dataset...", flush=True)
    df = data_loader.get_clean_data(subset=subset, data_dir=data_dir)

    if df is None or df.empty:
        raise ValueError("Dataset non caricato o vuoto.")

    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne", flush=True)
    print(f"Motori unici: {df['engine_id'].nunique()}", flush=True)

    if max_rows is not None and len(df) > max_rows:
        print(f"[INFO] Campionamento veloce: tengo {max_rows} righe su {len(df)}", flush=True)
        # campionamento ordinato per non distruggere completamente la struttura temporale
        df = df.sort_values(["engine_id", "cycle"]).groupby("engine_id", group_keys=False).head(
            max(1, max_rows // df["engine_id"].nunique())
        )
        print(f"Dataset dopo campionamento: {df.shape[0]} righe", flush=True)

    print("[2/6] Annotazione con Knowledge Base...", flush=True)
    if skip_kb:
        print("[INFO] skip_kb=True: creo colonne KB a False per test veloce.", flush=True)
        for col in [
            "kb_thermal_stress",
            "kb_pressure_instability",
            "kb_adverse_trend",
            "kb_degradation_evidence",
        ]:
            if col not in df.columns:
                df[col] = False
    else:
        kb_start = time.perf_counter()
        df = logic_engine.annotate_with_kb(df)
        kb_time = time.perf_counter() - kb_start
        print(f"Annotazione KB completata in {kb_time:.2f}s", flush=True)

    target = "failure_risk"
    if target not in df.columns:
        raise ValueError(f"Colonna target mancante: {target}")

    if df[target].nunique() < 2:
        raise ValueError("Target con una sola classe: confronto non significativo.")

    feature_cols = [
        "operating_regime",
        "sensor_anomaly_level",
        "thermal_state",
        "pressure_state",
        "rotation_state",
        "trend_state",
        "kb_thermal_stress",
        "kb_pressure_instability",
        "kb_adverse_trend",
        "kb_degradation_evidence",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne feature mancanti: {missing}")

    X = df[feature_cols].astype(str)
    y = df[target].astype(str).values
    groups = df["engine_id"].values

    print("[3/6] Statistiche dataset finale...", flush=True)
    print(f"Righe usate: {len(df)}", flush=True)
    print(f"Feature usate: {len(feature_cols)} -> {feature_cols}", flush=True)
    print("Distribuzione target:", flush=True)
    print(df[target].value_counts().to_string(), flush=True)
    print("Motori per GroupKFold:", len(np.unique(groups)), flush=True)

    n_groups = len(np.unique(groups))
    if n_groups < k:
        raise ValueError(f"Impossibile usare k={k}: motori disponibili={n_groups}")

    cv = GroupKFold(n_splits=k)

    models = {
        "Baseline Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=5,
        ),
        "Gaussian Naive Bayes": GaussianNB(),
    }

    results = {}

    print("[4/6] Inizio cross-validation manuale...", flush=True)
    print("-" * 90, flush=True)

    for model_idx, (name, model) in enumerate(models.items(), start=1):
        model_start = time.perf_counter()

        print(f"\n[{model_idx}/{len(models)}] Modello: {name}", flush=True)

        fold_accuracy = []
        fold_balanced_accuracy = []
        fold_f1_macro = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            fold_start = time.perf_counter()

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            train_groups = len(np.unique(groups[train_idx]))
            test_groups = len(np.unique(groups[test_idx]))

            print(
                f"  Fold {fold_idx}/{k} | "
                f"train={len(train_idx)} righe ({train_groups} motori), "
                f"test={len(test_idx)} righe ({test_groups} motori)...",
                flush=True,
            )

            pipeline = _make_pipeline(model, feature_cols)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            fold_accuracy.append(acc)
            fold_balanced_accuracy.append(bal_acc)
            fold_f1_macro.append(f1)

            fold_time = time.perf_counter() - fold_start

            print(
                f"    -> acc={acc:.3f}, bal_acc={bal_acc:.3f}, "
                f"f1_macro={f1:.3f} | {fold_time:.2f}s",
                flush=True,
            )

        model_time = time.perf_counter() - model_start

        results[name] = {
            "accuracy_mean": float(np.mean(fold_accuracy)),
            "accuracy_std": float(np.std(fold_accuracy, ddof=1)),
            "balanced_accuracy_mean": float(np.mean(fold_balanced_accuracy)),
            "balanced_accuracy_std": float(np.std(fold_balanced_accuracy, ddof=1)),
            "f1_macro_mean": float(np.mean(fold_f1_macro)),
            "f1_macro_std": float(np.std(fold_f1_macro, ddof=1)),
            "runtime_seconds": float(model_time),
        }

        print(
            f"  COMPLETATO {name} in {model_time:.2f}s | "
            f"F1={_format_mean_std(fold_f1_macro)}, "
            f"Balanced Accuracy={_format_mean_std(fold_balanced_accuracy)}, "
            f"Accuracy={_format_mean_std(fold_accuracy)}",
            flush=True,
        )

    print("\n[5/6] Tabella finale", flush=True)
    print("=" * 90)
    print("| Modello | F1-Macro | Balanced Accuracy | Accuracy | Runtime |")
    print("|---|---:|---:|---:|---:|")

    rows = []
    for name, stats in results.items():
        f1_str = f"{stats['f1_macro_mean']:.3f} ± {stats['f1_macro_std']:.3f}"
        bal_str = f"{stats['balanced_accuracy_mean']:.3f} ± {stats['balanced_accuracy_std']:.3f}"
        acc_str = f"{stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}"
        runtime_str = f"{stats['runtime_seconds']:.2f}s"

        print(f"| {name} | {f1_str} | {bal_str} | {acc_str} | {runtime_str} |")

        rows.append({
            "model": name,
            "f1_macro": f1_str,
            "balanced_accuracy": bal_str,
            "accuracy": acc_str,
            "f1_macro_mean": stats["f1_macro_mean"],
            "f1_macro_std": stats["f1_macro_std"],
            "balanced_accuracy_mean": stats["balanced_accuracy_mean"],
            "balanced_accuracy_std": stats["balanced_accuracy_std"],
            "accuracy_mean": stats["accuracy_mean"],
            "accuracy_std": stats["accuracy_std"],
            "runtime_seconds": stats["runtime_seconds"],
        })

    print("\n[6/6] Salvataggio CSV...", flush=True)
    os.makedirs("results", exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_path = f"results/model_comparison_{subset}.csv"
    out_df.to_csv(out_path, index=False)

    total_time = time.perf_counter() - total_start

    print(f"CSV salvato in: {out_path}", flush=True)
    print(f"Tempo totale: {total_time:.2f}s", flush=True)
    print("=" * 90)

    return results


def main():
    parser = argparse.ArgumentParser(description="KARE - Confronto baseline supervisionate")
    parser.add_argument("--subset", default=config.DEFAULT_SUBSET)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Usa solo un sottoinsieme di righe per test veloce. Non usare per risultati finali.",
    )

    parser.add_argument(
        "--skip-kb",
        action="store_true",
        help="Salta annotate_with_kb per test veloce. Non usare per risultati finali.",
    )

    args = parser.parse_args()

    run_model_comparison(
        subset=args.subset,
        data_dir=args.data_dir,
        k=args.k,
        max_rows=args.max_rows,
        skip_kb=args.skip_kb,
    )


if __name__ == "__main__":
    main()