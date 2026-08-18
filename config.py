"""
config.py - Configurazione centralizzata per il framework KARE applicato
alla Manutenzione Predittiva di Turbine Eoliche (Wind Turbine PHM).
"""

from pathlib import Path

# ==========================================
# 1. PERCORSI E DIRECTORY
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "WindTurbineData"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures_doc"
DOCS_DIR = BASE_DIR / "docs"

for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Dataset
TRAIN_DATA_FILE = DATA_DIR / "train_wind_turbines.csv"
TEST_DATA_FILE = DATA_DIR / "test_wind_turbines.csv"

# ==========================================
# 2. DEFINIZIONE FEATURE E SENSORI SCADA
# ==========================================
ID_COL = "turbine_id"
TIME_COL = "operating_hours"
RUL_COL = "RUL"

# Condizioni ambientali / operative
OPERATIONAL_SETTINGS = [
    "wind_speed_ms",
    "ambient_temp_c",
    "wind_direction_deg"
]

# Sensori di stato meccanico ed elettrico
SENSORS = [
    "gearbox_oil_temp_c",
    "gearbox_bearing_temp_c",
    "generator_winding_temp_c",
    "generator_rpm",
    "rotor_rpm",
    "blade_pitch_angle_deg",
    "hydraulic_pressure_bar",
    "active_power_kw"
]

ALL_FEATURES = OPERATIONAL_SETTINGS + SENSORS

# ==========================================
# 3. SOGLIE FISICHE DI ALLARME (Knowledge Base)
# ==========================================
PHYSICAL_THRESHOLDS = {
    # Temperature critiche (°C)
    "gearbox_oil_max_safe": 80.0,
    "gearbox_oil_alarm": 95.0,
    "bearing_temp_alarm": 90.0,
    "generator_winding_alarm": 120.0,
    
    # Pressione idraulica (bar)
    "hydraulic_min_safe": 140.0,
    "hydraulic_nominal": 180.0,
    "hydraulic_max_safe": 210.0,
    
    # Efficienza / Rapporti critici
    "min_power_curve_efficiency": 0.65,  # Potenza erogata rispetto alla curva teorica di potenza
    "max_rpm_overspeed": 1850.0          # Max RPM generatore prima del blocco di sicurezza
}

# ==========================================
# 4. STATI DI DEGRADO E SALUTE (Health Status)
# ==========================================
HEALTH_STATES = {
    "HEALTHY": 0,    # Funzionamento regolare (RUL > 300 ore)
    "WARNING": 1,    # Usura iniziale / Deriva termica (100 < RUL <= 300 ore)
    "CRITICAL": 2    # Guasto imminente (RUL <= 100 ore)
}

RUL_EARLY_CUTOFF = 125.0  # Piece-wise linear RUL clipping (come in NASA C-MAPSS)
CRITICAL_RUL_THRESHOLD = 100.0
WARNING_RUL_THRESHOLD = 300.0

# ==========================================
# 5. FEATURE ENGINEERING (Finestre temporali)
# ==========================================
ROLLING_WINDOW_SIZE = 12  # Finestra mobile per media e deviazione standard (es. 12 step = 2 ore se a 10 min)
ZSCORE_THRESHOLD = 2.5    # Soglia anomalia statistica su finestre mobili

# ==========================================
# 6. PARAMETRI OTTIMIZZATORE CSP (Manutenzione)
# ==========================================
CSP_CONFIG = {
    "planning_horizon_days": 14,          # Orizzonte temporale di pianificazione
    "max_crews_available": 3,              # Squadre di tecnici specializzati
    "max_hours_per_shift": 8,              # Ore max lavorative per turno
    "weather_wind_speed_limit_ms": 12.0,   # Limite vento per salita in navicella in sicurezza
    "cost_preventive_intervention": 2500,  # Costo intervento pianificato (€)
    "cost_corrective_intervention": 15000, # Costo intervento per fermo a guasto (€)
    "cost_power_loss_per_hour": 350        # Perdita economica media per ora di fermo (€/h)
}