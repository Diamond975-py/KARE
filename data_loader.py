"""
data_loader.py - Modulo di caricamento, preprocessing, calcolo della RUL
e generazione dati per il monitoraggio predittivo di turbine eoliche.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
import config

def generate_synthetic_wind_turbine_data(
    num_turbines: int = 30,
    max_operating_hours: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera un dataset SCADA sintetico realistico con pattern di degrado
    progressivo per moltiplicatore, cuscinetti e sistema idraulico.
    """
    np.random.seed(seed)
    records = []
    
    for t_id in range(1, num_turbines + 1):
        turbine_name = f"WTG_{t_id:02d}"
        lifetime = np.random.randint(250, max_operating_hours)
        
        # Stato iniziale dei componenti (piccole variazioni costruttive)
        gearbox_base_temp = np.random.uniform(60.0, 68.0)
        bearing_base_temp = np.random.uniform(55.0, 63.0)
        hydraulic_base_press = np.random.uniform(180.0, 190.0)
        
        for hour in range(1, lifetime + 1):
            # Profilo di vento con fluttuazioni stocastiche
            wind_speed = np.clip(np.random.normal(loc=9.0, scale=2.5), 3.0, 25.0)
            ambient_temp = np.random.normal(loc=18.0, scale=4.0)
            wind_direction = (np.random.normal(loc=180.0, scale=15.0) + (hour * 0.1)) % 360
            
            # Dinamica di rotazione e potenza in base al vento
            rotor_rpm = np.clip(wind_speed * 1.4 + np.random.normal(0, 0.2), 6.0, 20.0)
            generator_rpm = rotor_rpm * 90.0  # Rapporto tipico moltiplicatore 1:90
            active_power = np.clip((wind_speed ** 3) * 2.2 + np.random.normal(0, 15), 0, 2000)
            blade_pitch = 0.0 if wind_speed < 12 else (wind_speed - 12) * 2.5
            
            # Dinamica di degrado non lineare (esponenziale verso fine vita)
            degradation_factor = (hour / lifetime) ** 3.5
            
            # Sensori con deriva termica e usura idraulica
            gearbox_oil_temp = gearbox_base_temp + (active_power / 100) * 0.8 + (degradation_factor * 35.0) + np.random.normal(0, 0.8)
            gearbox_bearing_temp = bearing_base_temp + (generator_rpm / 300) * 1.1 + (degradation_factor * 32.0) + np.random.normal(0, 0.7)
            generator_winding_temp = ambient_temp + 45.0 + (active_power / 80) * 1.5 + (degradation_factor * 40.0) + np.random.normal(0, 1.0)
            hydraulic_pressure = hydraulic_base_press - (degradation_factor * 45.0) + np.random.normal(0, 1.5)
            
            records.append({
                config.ID_COL: turbine_name,
                config.TIME_COL: hour,
                "wind_speed_ms": round(wind_speed, 2),
                "ambient_temp_c": round(ambient_temp, 2),
                "wind_direction_deg": round(wind_direction, 2),
                "gearbox_oil_temp_c": round(gearbox_oil_temp, 2),
                "gearbox_bearing_temp_c": round(gearbox_bearing_temp, 2),
                "generator_winding_temp_c": round(generator_winding_temp, 2),
                "generator_rpm": round(generator_rpm, 2),
                "rotor_rpm": round(rotor_rpm, 2),
                "blade_pitch_angle_deg": round(blade_pitch, 2),
                "hydraulic_pressure_bar": round(hydraulic_pressure, 2),
                "active_power_kw": round(active_power, 2)
            })
            
    df = pd.DataFrame(records)
    return df


def calculate_rul(df: pd.DataFrame, early_cutoff: Optional[float] = config.RUL_EARLY_CUTOFF) -> pd.DataFrame:
    """
    Calcola la RUL (Remaining Useful Life) in ore per ogni turbina.
    RUL(t) = Max_Hours(Turbina) - Current_Hour(t)
    Applica opzionalmente il piecewise linear clipping.
    """
    df_processed = df.copy()
    
    # Trova il ciclo massimo per ciascuna turbina
    max_cycle_per_unit = df_processed.groupby(config.ID_COL)[config.TIME_COL].transform("max")
    
    # RUL lineare
    df_processed[config.RUL_COL] = max_cycle_per_unit - df_processed[config.TIME_COL]
    
    # RUL clipped (Early Cutoff) per stabilizzare l'apprendimento
    if early_cutoff is not None:
        df_processed["RUL_clipped"] = df_processed[config.RUL_COL].clip(upper=early_cutoff)
    else:
        df_processed["RUL_clipped"] = df_processed[config.RUL_COL]
        
    # Assegna lo stato di salute discreto
    def assign_health_state(rul):
        if rul <= config.CRITICAL_RUL_THRESHOLD:
            return config.HEALTH_STATES["CRITICAL"]
        elif rul <= config.WARNING_RUL_THRESHOLD:
            return config.HEALTH_STATES["WARNING"]
        return config.HEALTH_STATES["HEALTHY"]
        
    df_processed["health_state"] = df_processed[config.RUL_COL].apply(assign_health_state)
    
    return df_processed


def extract_rolling_features(df: pd.DataFrame, window_size: int = config.ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Estrae statistiche a finestra mobile (media, std) e z-score
    per catturare derive e anomalie transitorie.
    """
    df_features = df.copy()
    
    for sensor in config.SENSORS:
        # Media mobile e deviazione standard per ciascuna turbina
        rolling_mean = df_features.groupby(config.ID_COL)[sensor].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        rolling_std = df_features.groupby(config.ID_COL)[sensor].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std().fillna(1e-4)
        )
        
        df_features[f"{sensor}_roll_mean"] = rolling_mean
        df_features[f"{sensor}_roll_std"] = rolling_std
        df_features[f"{sensor}_zscore"] = (df_features[sensor] - rolling_mean) / rolling_std
        
    return df_features


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Carica i dati SCADA dal disco o genera un dataset sintetico se non presente.
    """
    path = data_path or config.TRAIN_DATA_FILE
    
    if not path.exists():
        print(f"[INFO] File {path} non trovato. Generazione dataset sintetico realistico in corso...")
        df_raw = generate_synthetic_wind_turbine_data(num_turbines=35)
        df_raw.to_csv(path, index=False)
        print(f"[OK] Dataset salvato con successo in: {path}")
    else:
        df_raw = pd.read_csv(path)
        
    # Calcolo RUL e feature engineering
    df_rul = calculate_rul(df_raw)
    df_final = extract_rolling_features(df_rul)
    
    return df_final


if __name__ == "__main__":
    # Test di funzionamento rapido
    print("--- Test Esecuzione data_loader.py ---")
    data = load_dataset()
    print(f"Dimensioni dataset: {data.shape}")
    print(f"Turbine totali: {data[config.ID_COL].nunique()}")
    print("Anteprima colonne e RUL:")
    print(data[[config.ID_COL, config.TIME_COL, "gearbox_oil_temp_c", "RUL", "health_state"]].head(8))