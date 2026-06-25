"""
experiment_runner.py - Esegue tutti gli esperimenti principali.

Produce risultati in formato dizionario e stampa metriche media ± deviazione
standard per:
- Knowledge Base;
- Rete Bayesiana;
- CSP.
"""

from __future__ import annotations

import argparse
import json

import bayesian_learner
import config
import csp_evaluation
import data_loader
import kb_evaluation


def run_all_experiments(subset=config.DEFAULT_SUBSET, data_dir=None, k=5):
    df = data_loader.get_clean_data(subset=subset, data_dir=data_dir)

    results = {
        "kb": kb_evaluation.evaluate_kb(df=df, k=k),
        "bayes": bayesian_learner.cross_validate_bayesian_network(df=df, k=k),
        "csp": csp_evaluation.evaluate_csp(df=df, k=k),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="KARE - Runner esperimenti")
    parser.add_argument("--subset", default=config.DEFAULT_SUBSET)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results = run_all_experiments(subset=args.subset, data_dir=args.data_dir, k=args.k)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Risultati salvati in {args.json_out}")


if __name__ == "__main__":
    main()
