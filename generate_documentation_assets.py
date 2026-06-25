"""
generate_documentation_figures.py

Genera le figure F1-F11 per la documentazione KARE in modo coerente
con la bozza della relazione.

Output:
    figures_doc/
        F1_architecture_kare.png
        F2_evidence_pipeline.png
        F3_rul_failure_distribution.png
        F4_rul_curves_multiple_engines.png
        F5_sensor_rolling_zscore.png
        F6_kb_rule_graph.png
        F7_bayesian_network_structure.png
        F8_bayes_comparison.png
        F9_csp_schema.png
        F10_csp_comparison.png
        F11_groupkfold_schema.png

Uso:
    python generate_documentation_figures.py --subset FD001 --data-dir data/CMAPSSData

Note:
    - Le figure F1, F2, F6, F7, F9, F11 sono schemi progettuali.
    - Le figure F3, F4, F5 usano i dati C-MAPSS reali.
    - Le figure F8 e F10 leggono CSV in results/ se presenti.
      Se i CSV non esistono, usano i valori riportati nella bozza della documentazione.
"""

import os
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import networkx as nx


# ============================================================
# CONFIGURAZIONE GENERALE
# ============================================================

FIG_DIR = Path("figures_doc")
RESULTS_DIR = Path("results")

RUL_THRESHOLDS = {
    "healthy_warning": 80,
    "warning_degraded": 40,
    "degraded_critical": 15,
}

WINDOW_SIZE = 20
BASELINE_WINDOW = 20

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Coerenti con le tabelle della bozza.
THERMAL_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_8", "sensor_11", "sensor_13", "sensor_15"]
PRESSURE_SENSORS = ["sensor_7", "sensor_11", "sensor_12", "sensor_20", "sensor_21"]
ROTATION_SENSORS = ["sensor_9", "sensor_14"]


# ============================================================
# UTILITY
# ============================================================

def ensure_dirs():
    FIG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def save_figure(filename: str):
    out_path = FIG_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] salvata {out_path}")


def wrap_label(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(label, width=width))


def add_box(ax, xy, text, width=0.62, height=0.08,
            facecolor="#EAF1FB", edgecolor="#1F2937",
            fontsize=11, weight="bold"):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
    )
    return box


def add_arrow(ax, start, end, color="#111827", lw=1.6, mutation_scale=18):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)


def format_bar_labels(ax, bars, fmt="{:.0f}", dy=3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ============================================================
# DATASET C-MAPSS
# ============================================================

def rul_to_class(rul: int) -> str:
    if rul > 80:
        return "healthy"
    if rul > 40:
        return "warning"
    if rul > 15:
        return "degraded"
    return "critical"


def rul_class_to_failure_risk(rul_class: str) -> str:
    mapping = {
        "healthy": "low",
        "warning": "medium",
        "degraded": "high",
        "critical": "critical",
    }
    return mapping[rul_class]


def load_cmapss_train(subset: str, data_dir: str) -> pd.DataFrame:
    """
    Carica direttamente train_FDxxx.txt senza dipendere dal data_loader del progetto.
    Questo rende lo script stabile anche se data_loader.py cambia.
    """
    file_path = Path(data_dir) / f"train_{subset}.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}\n"
            f"Controlla che la cartella contenga train_{subset}.txt"
        )

    columns = (
        ["engine_id", "cycle"]
        + OP_COLUMNS
        + SENSOR_COLUMNS
    )

    # I file C-MAPSS hanno spazi multipli e spesso colonne vuote finali.
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python"
    )

    # Tieni solo le prime 26 colonne: id, cycle, 3 settings, 21 sensors.
    df = df.iloc[:, :26]
    df.columns = columns

    df["subset"] = subset
    df["record_id"] = (
        df["subset"].astype(str)
        + "_E" + df["engine_id"].astype(int).astype(str)
        + "_C" + df["cycle"].astype(int).astype(str)
    )

    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    df["rul_class"] = df["rul"].apply(rul_to_class)
    df["failure_risk"] = df["rul_class"].apply(rul_class_to_failure_risk)
    df["urgent_label"] = (df["rul"] <= 40).astype(int)

    return df


def choose_representative_engines(df: pd.DataFrame, n: int = 4):
    """
    Sceglie motori con durate diverse, così il grafico F4 non mostra linee quasi uguali.
    """
    lengths = df.groupby("engine_id")["cycle"].max().sort_values()
    if len(lengths) <= n:
        return list(lengths.index)

    positions = np.linspace(0, len(lengths) - 1, n).astype(int)
    return list(lengths.iloc[positions].index)


def choose_engine_for_sensor_plot(df: pd.DataFrame) -> int:
    """
    Sceglie un motore lungo abbastanza da rendere leggibile rolling/z-score.
    """
    lengths = df.groupby("engine_id")["cycle"].max().sort_values(ascending=False)
    return int(lengths.index[0])


def compute_rolling_and_zscore(engine_df: pd.DataFrame, sensor: str):
    e = engine_df.sort_values("cycle").copy()

    rolling_mean = e[sensor].rolling(WINDOW_SIZE, min_periods=1).mean()

    baseline = e[sensor].iloc[:BASELINE_WINDOW]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std(ddof=0)

    if baseline_std == 0 or np.isnan(baseline_std):
        baseline_std = 1.0

    zscore = (e[sensor] - baseline_mean) / baseline_std

    return e["cycle"], e[sensor], rolling_mean, zscore


# ============================================================
# F1 - ARCHITETTURA GENERALE
# ============================================================

def figure_f1_architecture():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.axis("off")

    steps = [
        "NASA C-MAPSS\nDataset",
        "Data Loader\n+ Feature Engineering",
        "Knowledge Base\npyDatalog",
        "Bayesian Network\npgmpy",
        "CSP Maintenance\nOptimizer",
        "Evaluation\n& Reporting",
    ]

    x = 0.5
    y0 = 0.88
    gap = 0.14
    width = 0.62
    height = 0.08

    for i, step in enumerate(steps):
        y = y0 - i * gap
        add_box(ax, (x, y), step, width=width, height=height)

        if i < len(steps) - 1:
            next_y = y0 - (i + 1) * gap
            add_arrow(
                ax,
                (x, y - height / 2 - 0.01),
                (x, next_y + height / 2 + 0.01),
            )

    ax.set_title(
        "Architettura generale del sistema KARE",
        fontsize=16,
        weight="bold",
        pad=18,
    )

    save_figure("F1_architecture_kare.png")


# ============================================================
# F2 - PIPELINE EVIDENZE
# ============================================================

def figure_f2_evidence_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    steps = [
        ("Dati grezzi\nC-MAPSS", "record numerici"),
        ("Preprocessing", "RUL, rolling,\nz-score"),
        ("Stati simbolici", "normal / mild /\nmoderate / severe"),
        ("Fatti KB", "thermal_state,\npressure_state"),
        ("Inferenze KB", "degradation,\nurgent"),
        ("Evidenze Bayes", "variabili discrete\nosservate"),
        ("FailureRisk", "distribuzione\nprobabilistica"),
        ("Candidati CSP", "motori urgenti\n+ deadline"),
        ("Piano manutentivo", "giorno / slot /\ntecnico / azione"),
    ]

    xs = np.linspace(0.07, 0.93, len(steps))
    y = 0.55

    for i, (title, subtitle) in enumerate(steps):
        text = f"{title}\n{subtitle}"
        add_box(
            ax,
            (xs[i], y),
            text,
            width=0.105,
            height=0.20,
            fontsize=8.5,
            facecolor="#EEF2FF" if i not in [3, 4] else "#DBEAFE",
        )

        if i < len(steps) - 1:
            add_arrow(ax, (xs[i] + 0.055, y), (xs[i + 1] - 0.055, y), lw=1.4, mutation_scale=15)

    ax.text(
        xs[3],
        0.25,
        "Livello simbolico / KB",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="#1D4ED8",
    )

    ax.set_title(
        "Pipeline di esecuzione e passaggio delle evidenze",
        fontsize=16,
        weight="bold",
        pad=14,
    )

    save_figure("F2_evidence_pipeline.png")


# ============================================================
# F3 - DISTRIBUZIONE RUL / FAILURE RISK
# ============================================================

def figure_f3_rul_failure_distribution(df: pd.DataFrame):
    rul_order = ["healthy", "warning", "degraded", "critical"]
    risk_order = ["low", "medium", "high", "critical"]

    rul_counts = df["rul_class"].value_counts().reindex(rul_order).fillna(0)
    risk_counts = df["failure_risk"].value_counts().reindex(risk_order).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bars1 = axes[0].bar(rul_counts.index, rul_counts.values, color=["#60A5FA", "#93C5FD", "#FDBA74", "#F87171"])
    axes[0].set_title("Distribuzione RULClass")
    axes[0].set_xlabel("Classe RUL")
    axes[0].set_ylabel("Numero di record")
    axes[0].tick_params(axis="x", rotation=0)
    format_bar_labels(axes[0], bars1, fmt="{:.0f}")

    bars2 = axes[1].bar(risk_counts.index, risk_counts.values, color=["#60A5FA", "#FBBF24", "#FB923C", "#EF4444"])
    axes[1].set_title("Distribuzione FailureRisk")
    axes[1].set_xlabel("Classe di rischio")
    axes[1].set_ylabel("Numero di record")
    axes[1].tick_params(axis="x", rotation=0)
    format_bar_labels(axes[1], bars2, fmt="{:.0f}")

    fig.suptitle("Distribuzione delle classi RUL e FailureRisk", fontsize=16, weight="bold")
    plt.tight_layout()

    save_figure("F3_rul_failure_distribution.png")


# ============================================================
# F4 - RUL PER PIÙ MOTORI CON SOGLIE
# ============================================================

def figure_f4_rul_curves(df: pd.DataFrame):
    engines = choose_representative_engines(df, n=4)

    plt.figure(figsize=(9, 5.4))

    for engine_id in engines:
        e = df[df["engine_id"] == engine_id].sort_values("cycle")
        plt.plot(e["cycle"], e["rul"], linewidth=2, label=f"Motore {engine_id}")

    plt.axhline(80, linestyle="--", linewidth=1.2, color="#2563EB", label="Soglia healthy/warning = 80")
    plt.axhline(40, linestyle="--", linewidth=1.2, color="#F59E0B", label="Soglia warning/degraded = 40")
    plt.axhline(15, linestyle="--", linewidth=1.2, color="#DC2626", label="Soglia degraded/critical = 15")

    plt.title("Andamento del RUL per motori rappresentativi", fontsize=15, weight="bold")
    plt.xlabel("Ciclo")
    plt.ylabel("Remaining Useful Life")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    save_figure("F4_rul_curves_multiple_engines.png")


# ============================================================
# F5 - SENSORI, ROLLING MEAN, Z-SCORE, SOGLIE
# ============================================================

def figure_f5_sensor_rolling_zscore(df: pd.DataFrame):
    engine_id = choose_engine_for_sensor_plot(df)
    e = df[df["engine_id"] == engine_id].sort_values("cycle")

    sensors = [
        ("sensor_2", "proxy termico"),
        ("sensor_7", "proxy pressione"),
    ]

    fig, axes = plt.subplots(len(sensors), 2, figsize=(13, 7.2), sharex=True)

    if len(sensors) == 1:
        axes = np.array([axes])

    for row_idx, (sensor, label) in enumerate(sensors):
        cycles, raw, rolling_mean, zscore = compute_rolling_and_zscore(e, sensor)

        ax_raw = axes[row_idx, 0]
        ax_z = axes[row_idx, 1]

        ax_raw.plot(cycles, raw, linewidth=1.2, alpha=0.55, label="Valore grezzo")
        ax_raw.plot(cycles, rolling_mean, linewidth=2.2, label=f"Rolling mean ({WINDOW_SIZE} cicli)")
        ax_raw.set_title(f"{sensor} ({label}) - valore e rolling mean")
        ax_raw.set_ylabel("Valore")
        ax_raw.grid(alpha=0.2)
        ax_raw.legend(fontsize=8)

        ax_z.plot(cycles, zscore, linewidth=1.6, label="z-score")
        ax_z.axhline(2, linestyle="--", linewidth=1.1, color="#F59E0B", label="|z| = 2")
        ax_z.axhline(-2, linestyle="--", linewidth=1.1, color="#F59E0B")
        ax_z.axhline(3, linestyle="--", linewidth=1.1, color="#DC2626", label="|z| = 3")
        ax_z.axhline(-3, linestyle="--", linewidth=1.1, color="#DC2626")
        ax_z.set_title(f"{sensor} ({label}) - z-score rispetto alla baseline")
        ax_z.set_ylabel("z-score")
        ax_z.grid(alpha=0.2)
        ax_z.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Ciclo")

    fig.suptitle(
        f"Andamento sensori, rolling mean e z-score - Motore {engine_id}",
        fontsize=16,
        weight="bold",
    )

    plt.tight_layout()

    save_figure("F5_sensor_rolling_zscore.png")


# ============================================================
# F6 - GRAFO KB COERENTE CON T7/T8/T9
# ============================================================

def figure_f6_kb_rule_graph():
    G = nx.DiGraph()

    edges = [
        ("thermal_state_fact", "thermal_stress"),
        ("pressure_state_fact", "pressure_instability"),
        ("rotation_state_fact", "rotation_instability"),
        ("trend_state_fact", "adverse_trend"),
        ("anomaly_level_fact", "multiple_sensor_anomaly"),

        ("thermal_stress", "degradation_evidence"),
        ("pressure_instability", "degradation_evidence"),
        ("rotation_instability", "degradation_evidence"),
        ("adverse_trend", "degradation_evidence"),
        ("multiple_sensor_anomaly", "degradation_evidence"),

        ("thermal_stress", "critical_engine"),
        ("pressure_instability", "critical_engine"),
        ("adverse_trend", "critical_engine"),
        ("multiple_sensor_anomaly", "critical_engine"),

        ("degradation_evidence", "urgent_maintenance"),
        ("adverse_trend", "urgent_maintenance"),
        ("critical_engine", "urgent_maintenance"),

        ("degradation_evidence", "needs_inspection"),
        ("urgent_maintenance", "needs_repair"),
        ("critical_engine", "needs_replacement"),

        ("urgent_maintenance", "maintenance_deadline"),
        ("critical_engine", "maintenance_deadline"),
        ("degradation_evidence", "maintenance_deadline"),
    ]

    G.add_edges_from(edges)

    pos = {
        # fatti estensionali
        "thermal_state_fact": (-3.0, 3.0),
        "pressure_state_fact": (-1.5, 3.0),
        "rotation_state_fact": (0.0, 3.0),
        "trend_state_fact": (1.5, 3.0),
        "anomaly_level_fact": (3.0, 3.0),

        # diagnosi intermedie
        "thermal_stress": (-3.0, 1.8),
        "pressure_instability": (-1.5, 1.8),
        "rotation_instability": (0.0, 1.8),
        "adverse_trend": (1.5, 1.8),
        "multiple_sensor_anomaly": (3.0, 1.8),

        # inferenze centrali
        "degradation_evidence": (-1.2, 0.55),
        "critical_engine": (1.2, 0.55),

        # decisione
        "urgent_maintenance": (0.0, -0.65),

        # azioni
        "needs_inspection": (-2.4, -1.85),
        "needs_repair": (0.0, -1.85),
        "needs_replacement": (2.4, -1.85),

        # deadline
        "maintenance_deadline": (0.0, -3.0),
    }

    node_colors = []
    for node in G.nodes:
        if node.endswith("_fact"):
            node_colors.append("#DBEAFE")
        elif node in {"degradation_evidence", "critical_engine", "urgent_maintenance"}:
            node_colors.append("#FDE68A")
        elif node.startswith("needs_") or node == "maintenance_deadline":
            node_colors.append("#DCFCE7")
        else:
            node_colors.append("#E5E7EB")

    plt.figure(figsize=(14, 9))

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=16,
        width=1.4,
        edge_color="#374151",
        alpha=0.85,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2900,
        node_color=node_colors,
        edgecolors="#111827",
        linewidths=1.2,
    )

    labels = {node: wrap_label(node, 18) for node in G.nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8.5,
        font_weight="bold",
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Fatti estensionali", markerfacecolor="#DBEAFE", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Inferenze centrali", markerfacecolor="#FDE68A", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Azioni / deadline", markerfacecolor="#DCFCE7", markersize=12),
    ]

    plt.legend(handles=legend_elements, loc="lower right")
    plt.title("Grafo delle regole della Knowledge Base", fontsize=16, weight="bold", pad=14)
    plt.axis("off")
    plt.tight_layout()

    save_figure("F6_kb_rule_graph.png")


# ============================================================
# F7 - RETE BAYESIANA COERENTE CON T11/T12
# ============================================================

def figure_f7_bayesian_network():
    G = nx.DiGraph()

    edges = [
        ("OperatingRegime", "SensorAnomalyLevel"),

        ("SensorAnomalyLevel", "ThermalState"),
        ("SensorAnomalyLevel", "PressureState"),
        ("SensorAnomalyLevel", "TrendState"),
        ("SensorAnomalyLevel", "FailureRisk"),

        ("ThermalState", "KB_ThermalStress"),
        ("PressureState", "KB_PressureInstability"),
        ("TrendState", "KB_AdverseTrend"),

        ("KB_ThermalStress", "KB_DegradationEvidence"),
        ("KB_PressureInstability", "KB_DegradationEvidence"),
        ("KB_AdverseTrend", "KB_DegradationEvidence"),

        ("KB_DegradationEvidence", "FailureRisk"),
    ]

    G.add_edges_from(edges)

    pos = {
        "OperatingRegime": (-3.0, 3.0),
        "SensorAnomalyLevel": (-3.0, 1.8),

        "ThermalState": (-1.6, 0.8),
        "PressureState": (0.0, 0.8),
        "TrendState": (1.6, 0.8),

        "KB_ThermalStress": (-1.6, -0.5),
        "KB_PressureInstability": (0.0, -0.5),
        "KB_AdverseTrend": (1.6, -0.5),

        "KB_DegradationEvidence": (0.0, -1.8),
        "FailureRisk": (0.0, -3.0),
    }

    node_colors = []
    for node in G.nodes:
        if node.startswith("KB_"):
            node_colors.append("#FEF3C7")
        elif node == "FailureRisk":
            node_colors.append("#FECACA")
        elif node in {"ThermalState", "PressureState", "TrendState"}:
            node_colors.append("#DBEAFE")
        else:
            node_colors.append("#E0E7FF")

    plt.figure(figsize=(11, 9))

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        width=1.6,
        edge_color="#374151",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3600,
        node_color=node_colors,
        edgecolors="#111827",
        linewidths=1.3,
    )

    labels = {node: wrap_label(node, 17) for node in G.nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Variabili sensoristiche", markerfacecolor="#DBEAFE", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Output Knowledge Base", markerfacecolor="#FEF3C7", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Target probabilistico", markerfacecolor="#FECACA", markersize=12),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.title("Struttura della rete bayesiana", fontsize=16, weight="bold", pad=14)
    plt.axis("off")
    plt.tight_layout()

    save_figure("F7_bayesian_network_structure.png")


# ============================================================
# F8 - CONFRONTO BAYES
# ============================================================

def load_bayes_results(subset: str) -> pd.DataFrame:
    """
    Legge results/bayes_cv_FD001.csv se presente.
    Altrimenti usa i valori già presenti nella bozza della documentazione.
    """
    path = RESULTS_DIR / f"bayes_cv_{subset}.csv"

    if path.exists():
        df = pd.read_csv(path)
        print(f"[INFO] Uso risultati Bayes reali da {path}")
        return df

    print("[WARN] results/bayes_cv non trovato: uso i valori della bozza documentale.")
    return pd.DataFrame([
        {
            "model": "Bayes only",
            "accuracy_mean": 0.78,
            "accuracy_std": 0.03,
            "balanced_accuracy_mean": 0.75,
            "balanced_accuracy_std": 0.04,
            "f1_macro_mean": 0.74,
            "f1_macro_std": 0.04,
            "brier_mean": 0.18,
            "brier_std": 0.02,
        },
        {
            "model": "KB + Bayes",
            "accuracy_mean": 0.83,
            "accuracy_std": 0.02,
            "balanced_accuracy_mean": 0.81,
            "balanced_accuracy_std": 0.03,
            "f1_macro_mean": 0.80,
            "f1_macro_std": 0.03,
            "brier_mean": 0.15,
            "brier_std": 0.02,
        },
    ])


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Nessuna colonna trovata tra: {candidates}")


def figure_f8_bayes_comparison(subset: str):
    df = load_bayes_results(subset)

    model_col = get_col(df, ["model", "Modello"])
    acc_col = get_col(df, ["accuracy_mean", "Accuracy_mean", "accuracy"])
    acc_std_col = get_col(df, ["accuracy_std", "Accuracy_std"])
    bal_col = get_col(df, ["balanced_accuracy_mean", "balanced_acc_mean", "Balanced Accuracy_mean"])
    bal_std_col = get_col(df, ["balanced_accuracy_std", "balanced_acc_std", "Balanced Accuracy_std"])
    f1_col = get_col(df, ["f1_macro_mean", "f1_mean", "F1 macro_mean"])
    f1_std_col = get_col(df, ["f1_macro_std", "f1_std", "F1 macro_std"])
    brier_col = get_col(df, ["brier_mean", "Brier score_mean"])
    brier_std_col = get_col(df, ["brier_std", "Brier score_std"])

    models = df[model_col].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x - width, df[acc_col], width, yerr=df[acc_std_col], capsize=4, label="Accuracy")
    axes[0].bar(x, df[bal_col], width, yerr=df[bal_std_col], capsize=4, label="Balanced Accuracy")
    axes[0].bar(x + width, df[f1_col], width, yerr=df[f1_std_col], capsize=4, label="F1 macro")
    axes[0].set_title("Metriche predittive")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    axes[1].bar(x, df[brier_col], width=0.45, yerr=df[brier_std_col], capsize=4)
    axes[1].set_title("Brier score multiclass\n(valore minore = migliore)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Brier score")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Confronto Bayes only vs KB + Bayes", fontsize=16, weight="bold")
    plt.tight_layout()

    save_figure("F8_bayes_comparison.png")


# ============================================================
# F9 - SCHEMA CSP
# ============================================================

def figure_f9_csp_schema():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    boxes = {
        "input": (0.12, 0.70, "Input\nKB + Bayes\nrischio, urgenza,\ndeadline"),
        "candidates": (0.34, 0.70, "Motori\ncandidati\nmax 8"),
        "variables": (0.56, 0.70, "Variabili CSP\nengine_i"),
        "domains": (0.78, 0.70, "Domini\n(giorno, slot,\ntecnico, azione)"),

        "constraints": (0.34, 0.34, "Hard constraint\ndeadline, budget,\ncapacità, skill"),
        "score": (0.56, 0.34, "Funzione score\npriorità rischio\n- costo"),
        "plan": (0.78, 0.34, "Piano\nmanutentivo\nordinato"),
    }

    for key, (x, y, text) in boxes.items():
        color = "#DBEAFE"
        if key in {"constraints"}:
            color = "#FDE68A"
        if key in {"plan"}:
            color = "#DCFCE7"

        add_box(
            ax,
            (x, y),
            text,
            width=0.18,
            height=0.18,
            facecolor=color,
            fontsize=10,
        )

    add_arrow(ax, (0.21, 0.70), (0.25, 0.70))
    add_arrow(ax, (0.43, 0.70), (0.47, 0.70))
    add_arrow(ax, (0.65, 0.70), (0.69, 0.70))

    add_arrow(ax, (0.78, 0.60), (0.78, 0.45))
    add_arrow(ax, (0.70, 0.34), (0.65, 0.34))
    add_arrow(ax, (0.47, 0.34), (0.43, 0.34))
    add_arrow(ax, (0.43, 0.34), (0.47, 0.34))
    add_arrow(ax, (0.65, 0.34), (0.69, 0.34))

    ax.text(
        0.56,
        0.52,
        "Il CSP esplora assegnazioni valide e seleziona il piano con score massimo",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
    )

    ax.set_title("Schema del CSP manutentivo", fontsize=16, weight="bold", pad=14)

    save_figure("F9_csp_schema.png")


# ============================================================
# F10 - CONFRONTO CSP
# ============================================================

def load_csp_results(subset: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"csp_evaluation_{subset}.csv"

    if path.exists():
        df = pd.read_csv(path)
        print(f"[INFO] Uso risultati CSP reali da {path}")
        return df

    print("[WARN] results/csp_evaluation non trovato: uso i valori della bozza documentale.")
    return pd.DataFrame([
        {
            "configuration": "CSP base",
            "deadline_satisfaction_mean": 0.84,
            "deadline_satisfaction_std": 0.03,
            "critical_coverage_mean": 0.79,
            "critical_coverage_std": 0.04,
            "total_cost_mean": 12600,
            "total_cost_std": 850,
            "runtime_mean": 1.8,
            "runtime_std": 0.3,
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
            "runtime_std": 0.4,
        },
    ])


def figure_f10_csp_comparison(subset: str):
    df = load_csp_results(subset)

    cfg_col = get_col(df, ["configuration", "Configurazione"])

    deadline_col = get_col(df, ["deadline_satisfaction_mean", "Deadline satisfaction_mean"])
    deadline_std_col = get_col(df, ["deadline_satisfaction_std", "Deadline satisfaction_std"])

    # Supporta sia critical_coverage sia urgent_coverage.
    coverage_col = get_col(df, ["critical_coverage_mean", "urgent_coverage_mean", "Critical coverage_mean"])
    coverage_std_col = get_col(df, ["critical_coverage_std", "urgent_coverage_std", "Critical coverage_std"])

    cost_col = get_col(df, ["total_cost_mean", "Costo totale_mean"])
    cost_std_col = get_col(df, ["total_cost_std", "Costo totale_std"])

    runtime_col = get_col(df, ["runtime_mean", "Runtime_mean"])
    runtime_std_col = get_col(df, ["runtime_std", "Runtime_std"])

    configs = df[cfg_col].tolist()
    x = np.arange(len(configs))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(x, df[deadline_col], yerr=df[deadline_std_col], capsize=4)
    axes[0, 0].set_title("Deadline satisfaction")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(configs)
    axes[0, 0].grid(axis="y", alpha=0.2)

    axes[0, 1].bar(x, df[coverage_col], yerr=df[coverage_std_col], capsize=4)
    axes[0, 1].set_title("Critical / urgent coverage")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(configs)
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[1, 0].bar(x, df[cost_col], yerr=df[cost_std_col], capsize=4)
    axes[1, 0].set_title("Costo totale medio")
    axes[1, 0].set_ylabel("Costo")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(configs)
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].bar(x, df[runtime_col], yerr=df[runtime_std_col], capsize=4)
    axes[1, 1].set_title("Runtime medio")
    axes[1, 1].set_ylabel("Secondi")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(configs)
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle("Confronto CSP base vs CSP risk-aware", fontsize=16, weight="bold")
    plt.tight_layout()

    save_figure("F10_csp_comparison.png")


# ============================================================
# F11 - GROUPKFOLD ANTI-LEAKAGE
# ============================================================

def figure_f11_groupkfold_schema():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    ax.set_title(
        "Schema GroupKFold per evitare data leakage tra cicli dello stesso motore",
        fontsize=15,
        weight="bold",
        pad=14,
    )

    # Area sinistra: split sbagliato.
    ax.text(0.25, 0.88, "Split casuale sulle righe\n(sbagliato)", ha="center", fontsize=13, weight="bold", color="#B91C1C")
    add_box(ax, (0.25, 0.68), "engine_1\ncicli 1-80\nTRAIN", width=0.22, height=0.14, facecolor="#DBEAFE")
    add_box(ax, (0.25, 0.48), "engine_1\ncicli 81-120\nTEST", width=0.22, height=0.14, facecolor="#FECACA")
    ax.text(
        0.25,
        0.28,
        "Lo stesso motore compare\nsia in train sia in test:\nstima troppo ottimistica.",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Area destra: GroupKFold corretto.
    ax.text(0.75, 0.88, "GroupKFold su engine_id\n(corretto)", ha="center", fontsize=13, weight="bold", color="#166534")
    add_box(ax, (0.75, 0.68), "engine_1\nTUTTI i cicli\nTRAIN", width=0.22, height=0.14, facecolor="#DBEAFE")
    add_box(ax, (0.75, 0.48), "engine_2\nTUTTI i cicli\nTEST", width=0.22, height=0.14, facecolor="#DCFCE7")
    ax.text(
        0.75,
        0.28,
        "Ogni motore appartiene\ninteramente a un solo fold:\nvalutazione più realistica.",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Separatore.
    ax.plot([0.5, 0.5], [0.18, 0.90], color="#9CA3AF", linewidth=1.2, linestyle="--")

    save_figure("F11_groupkfold_schema.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Genera figure documentazione KARE F1-F11")
    parser.add_argument("--subset", default="FD001", help="Subset C-MAPSS, es. FD001")
    parser.add_argument("--data-dir", default="data/CMAPSSData", help="Cartella contenente train_FD001.txt ecc.")
    args = parser.parse_args()

    ensure_dirs()

    print(f"[INFO] Caricamento dataset {args.subset} da {args.data_dir}")
    df = load_cmapss_train(subset=args.subset, data_dir=args.data_dir)

    print(f"[INFO] Record caricati: {len(df)}")
    print(f"[INFO] Motori: {df['engine_id'].nunique()}")
    print(f"[INFO] RULClass:\n{df['rul_class'].value_counts()}")

    figure_f1_architecture()
    figure_f2_evidence_pipeline()
    figure_f3_rul_failure_distribution(df)
    figure_f4_rul_curves(df)
    figure_f5_sensor_rolling_zscore(df)
    figure_f6_kb_rule_graph()
    figure_f7_bayesian_network()
    figure_f8_bayes_comparison(args.subset)
    figure_f9_csp_schema()
    figure_f10_csp_comparison(args.subset)
    figure_f11_groupkfold_schema()

    print("\n=== COMPLETATO ===")
    print(f"Figure generate in: {FIG_DIR.resolve()}")
    print("\nMappa figure per la documentazione:")
    print("F1  -> F1_architecture_kare.png")
    print("F2  -> F2_evidence_pipeline.png")
    print("F3  -> F3_rul_failure_distribution.png")
    print("F4  -> F4_rul_curves_multiple_engines.png")
    print("F5  -> F5_sensor_rolling_zscore.png")
    print("F6  -> F6_kb_rule_graph.png")
    print("F7  -> F7_bayesian_network_structure.png")
    print("F8  -> F8_bayes_comparison.png")
    print("F9  -> F9_csp_schema.png")
    print("F10 -> F10_csp_comparison.png")
    print("F11 -> F11_groupkfold_schema.png")


if __name__ == "__main__":
    main()