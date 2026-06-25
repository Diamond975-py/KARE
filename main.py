"""
main.py - CLI principale di KARE.

Esempi:
    python main.py --subset FD001 --run-analysis
    python main.py --subset FD001 --cross-validate --k 5
    python main.py --subset FD001 --evaluate-kb
    python main.py --subset FD001 --evaluate-csp
    python main.py --subset FD001 --compare-models
    python main.py --subset FD001 --engine-id 42
"""

from __future__ import annotations

import argparse
from typing import Optional

import bayesian_learner
import config
import csp_evaluation
import cv_balanced_validation
import data_loader
import kb_evaluation
import logic_engine
import maintenance_optimizer


def _print_dataset_summary(df):
    print("\nDataset caricato")
    print("----------------")
    print(f"Righe: {len(df)}")
    print(f"Motori: {df['engine_id'].nunique()}")
    print(f"Cicli: min={df['cycle'].min()}, max={df['cycle'].max()}")
    if "failure_risk" in df.columns:
        print("\nDistribuzione FailureRisk:")
        print(df["failure_risk"].value_counts().sort_index().to_string())


def _print_engine_report(df, engine_id: int):
    latest = data_loader.get_latest_engine_state(df)
    row_df = latest[latest["engine_id"] == int(engine_id)].copy()
    if row_df.empty:
        print(f"Motore {engine_id} non trovato.")
        return

    full_model = bayesian_learner.train_model(df=df)
    row_df = logic_engine.annotate_with_kb(row_df)
    row_df = bayesian_learner.predict_dataframe(row_df, model=full_model)
    row = row_df.iloc[0]

    print("\nReport motore")
    print("-------------")
    print(f"Engine ID: {row['engine_id']}")
    print(f"Ultimo ciclo: {row['cycle']}")
    if "rul" in row:
        print(f"RUL reale nel training: {row['rul']} ({row['rul_class']})")
    print(f"Anomaly level: {row['sensor_anomaly_level']}")
    print(f"Thermal state: {row['thermal_state']}")
    print(f"Pressure state: {row['pressure_state']}")
    print(f"Trend state: {row['trend_state']}")
    print(f"KB degradation evidence: {bool(row['kb_degradation_evidence'])}")
    print(f"KB urgent maintenance: {bool(row['kb_urgent_maintenance'])}")
    print(f"KB action: {row['kb_recommended_action']}")
    print(f"Predicted FailureRisk: {row['predicted_FailureRisk']}")

    prob_cols = [c for c in row_df.columns if c.startswith("prob_FailureRisk_")]
    if prob_cols:
        print("\nDistribuzione probabilistica:")
        for col in sorted(prob_cols):
            print(f" - {col.replace('prob_FailureRisk_', '')}: {float(row[col]):.3f}")


def run_analysis(subset: str, data_dir: Optional[str]):
    print("=" * 60)
    print("KARE - Knowledge-based Aircraft Risk & Engine Maintenance")
    print("=" * 60)

    df = data_loader.get_clean_data(subset=subset, data_dir=data_dir)
    _print_dataset_summary(df)

    print("\n[1/4] Inizializzazione Knowledge Base...")
    annotated = logic_engine.annotate_with_kb(df)
    print(f"Record con evidenza di degrado KB: {annotated['kb_degradation_evidence'].mean():.2%}")
    print(f"Record con manutenzione urgente KB: {annotated['kb_urgent_maintenance'].mean():.2%}")

    print("\n[2/4] Addestramento rete bayesiana...")
    model = bayesian_learner.train_model(df=annotated)

    print("\n[3/4] Predizione rischio su ultimo stato di ogni motore...")
    latest = data_loader.get_latest_engine_state(annotated)
    latest = logic_engine.annotate_with_kb(latest)
    latest = bayesian_learner.predict_dataframe(latest, model=model)

    columns = [
        "engine_id",
        "cycle",
        "predicted_FailureRisk",
        "sensor_anomaly_level",
        "kb_degradation_evidence",
        "kb_urgent_maintenance",
        "kb_recommended_action",
    ]
    print(latest[columns].head(15).to_string(index=False))

    print("\n[4/4] Pianificazione manutenzione con CSP...")
    plans = maintenance_optimizer.find_maintenance_schedule(engine_states=latest)
    if not plans:
        print("Nessun piano manutentivo necessario o nessuna soluzione CSP trovata.")
        return

    best = plans[0]
    print(f"\nMiglior piano - Score: {best['score']:.3f}")
    print(f"Costo totale: {best['total_cost']}")
    print(f"Deadline satisfaction: {best['deadline_satisfaction_rate']:.2%}")
    print(f"Runtime CSP: {best['runtime_seconds']:.4f}s")

    print("\nSchedule:")
    for item in best["schedule"]:
        print(
            f" - Giorno {item['scheduled_day']} {item['slot']}: "
            f"engine {item['engine_id']} | {item['action']} | "
            f"{item['technician']} | risk={item['risk']} | costo={item['cost']}"
        )


def main():
    parser = argparse.ArgumentParser(description="KARE - Sistema Knowledge-Based per Manutenzione Predittiva")
    parser.add_argument("--subset", default=config.DEFAULT_SUBSET, help="FD001, FD002, FD003 o FD004")
    parser.add_argument("--data-dir", default=None, help="Percorso cartella CMAPSSData")
    parser.add_argument("--k", type=int, default=5, help="Numero fold GroupKFold")

    parser.add_argument("--run-analysis", action="store_true", help="Esegue pipeline KB + Bayes + CSP")
    parser.add_argument("--cross-validate", action="store_true", help="Valuta rete bayesiana")
    parser.add_argument("--evaluate-kb", action="store_true", help="Valuta Knowledge Base")
    parser.add_argument("--evaluate-csp", action="store_true", help="Valuta CSP")
    parser.add_argument("--compare-models", action="store_true", help="Confronta baseline supervisionate")
    parser.add_argument("--engine-id", type=int, default=None, help="Mostra report su un motore specifico")

    args = parser.parse_args()

    try:
        if args.cross_validate:
            df = data_loader.get_clean_data(subset=args.subset, data_dir=args.data_dir)
            bayesian_learner.cross_validate_bayesian_network(df=df, k=args.k)
            return

        if args.evaluate_kb:
            kb_evaluation.evaluate_kb(subset=args.subset, data_dir=args.data_dir, k=args.k)
            return

        if args.evaluate_csp:
            csp_evaluation.evaluate_csp(subset=args.subset, data_dir=args.data_dir, k=args.k)
            return

        if args.compare_models:
            cv_balanced_validation.run_model_comparison(subset=args.subset, data_dir=args.data_dir, k=args.k)
            return

        if args.engine_id is not None:
            df = data_loader.get_clean_data(subset=args.subset, data_dir=args.data_dir)
            _print_engine_report(df, args.engine_id)
            return

        # Default sensato: analisi completa.
        run_analysis(args.subset, args.data_dir)

    except FileNotFoundError as exc:
        print("\nErrore dataset:")
        print(exc)
        print("\nSuggerimento: metti i file in data/CMAPSSData oppure usa --data-dir /percorso/CMAPSSData")


if __name__ == "__main__":
    main()
