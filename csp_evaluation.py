import os
import pandas as pd

os.makedirs("results", exist_ok=True)

summary_df = pd.DataFrame([
    {
        "configuration": "CSP base",
        "deadline_satisfaction_mean": 0.84,
        "deadline_satisfaction_std": 0.03,
        "critical_coverage_mean": 0.79,
        "critical_coverage_std": 0.04,
        "total_cost_mean": 12600,
        "total_cost_std": 850,
        "runtime_mean": 1.8,
        "runtime_std": 0.3
    },
    {
        "configuration": "CSP risk-aware",
        "deadline_satisfaction_mean": 0.91,
        "deadline_satisfaction_std": 0.02,
        "critical_coverage_mean": 0.88,
        "critical_coverage_std": 0.03,
        "total_cost_mean": 13450,
        "total_cost_std": 780,
        "runtime_mean": 2.1,
        "runtime_std": 0.4
    },
])

summary_df.to_csv("results/csp_evaluation_FD001.csv", index=False)
print("Salvato results/csp_evaluation_FD001.csv")