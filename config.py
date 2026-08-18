"""
Configurazione centrale di KARE.

KARE = Knowledge-based Aircraft Risk & Engine Maintenance
Dominio: diagnosi del degrado e pianificazione manutentiva su NASA C-MAPSS.

Le soglie sono volutamente esplicite perché devono essere motivate nella relazione.
"""

from pathlib import Path

# Dataset --------------------------------------------------------------------
DEFAULT_SUBSET = "FD001"
DATA_DIR_CANDIDATES = [
    Path("data") / "CMAPSSData",
    Path("data") / "CMAPSS",
    Path("CMAPSSData"),
    Path("CMAPSS"),
]

# Feature engineering ---------------------------------------------------------
WINDOW_SIZE = 20
BASELINE_WINDOW = 20
MAX_RUL_CAP = 125

# Classi RUL: il target principale deriva dal Remaining Useful Life.
# RUL > 80       -> healthy
# 40 < RUL <= 80 -> warning
# 15 < RUL <= 40 -> degraded
# RUL <= 15      -> critical
RUL_WARNING_THRESHOLD = 80
RUL_DEGRADED_THRESHOLD = 40
RUL_CRITICAL_THRESHOLD = 15

# Soglie per anomalie normalizzate rispetto alla baseline iniziale del motore.
Z_ANOMALY = 2.0
Z_CRITICAL = 3.0
TREND_EPS = 0.02

# Gruppi euristici di sensori.
# C-MAPSS espone sensori numerici; qui li raggruppiamo come proxy diagnostici,
# senza attribuire un significato fisico troppo specifico al singolo sensore.
THERMAL_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_8", "sensor_11", "sensor_13", "sensor_15"]
PRESSURE_SENSORS = ["sensor_7", "sensor_11", "sensor_12", "sensor_20", "sensor_21"]
ROTATION_SENSORS = ["sensor_9", "sensor_14"]

# Rete Bayesiana --------------------------------------------------------------
BAYES_PSEUDO_COUNTS = 1
BAYES_TARGET = "FailureRisk"

# CSP manutentivo -------------------------------------------------------------
CSP_DAYS = 7
CSP_SLOTS = ["morning", "afternoon"]
CSP_TECHNICIANS = {
    "tech_A": {"inspection", "repair"},
    "tech_B": {"inspection", "repair", "replacement"},
    "tech_C": {"inspection", "repair", "replacement"},
}
CSP_ACTION_COSTS = {
    "inspection": 1200,
    "repair": 4200,
    "replacement": 9000,
}
CSP_DAILY_BUDGET = 16000
CSP_MAX_ENGINES_PER_DAY = 4
CSP_MAX_CANDIDATES = 8
CSP_MAX_SOLUTIONS = 10
