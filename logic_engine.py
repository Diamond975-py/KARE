"""
logic_engine.py - Knowledge-Based System & Logic Inference Engine
per la diagnostica predittiva di guasti e usura in turbine eoliche.

Ruoli e caratteristiche del modulo:

Regole di Primo Ordine Esplicite: Cattura combinazioni fisiche reali (es. attrito cuscinetti + derivata termica = guasto al moltiplicatore).

Diagnosi Multi-Sottosistema: Distingue guasti al circuito idraulico del passo pale, al generatore elettrico e alla scatola ingranaggi.

Integrazione Z-Score: Riconosce le derive termiche repentine rispetto alla baseline mobile.

"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import config

class WindTurbineKnowledgeBase:
    """
    Knowledge Base contenente i fatti di dominio, le ontologie di allarme
    e le regole logiche di inferenza per i sottosistemi dell'aerogeneratore.
    """

    def __init__(self):
        self.thresholds = config.PHYSICAL_THRESHOLDS
        self.health_states = config.HEALTH_STATES

    def evaluate_telemetry_facts(self, row: pd.Series) -> Dict[str, bool]:
        """
        Estrae i fatti atomici (booleani) dalla telemetria SCADA istantanea
        e dalle metriche di rolling z-score.
        """
        facts = {}

        # 1. Fatti termici - Moltiplicatore di giri (Gearbox)
        facts["gearbox_oil_high"] = bool(row["gearbox_oil_temp_c"] >= self.thresholds["gearbox_oil_max_safe"])
        facts["gearbox_oil_critical"] = bool(row["gearbox_oil_temp_c"] >= self.thresholds["gearbox_oil_alarm"])
        facts["bearing_temp_critical"] = bool(row["gearbox_bearing_temp_c"] >= self.thresholds["bearing_temp_alarm"])

        # 2. Fatti termici - Generatore elettrico
        facts["generator_winding_critical"] = bool(
            row["generator_winding_temp_c"] >= self.thresholds["generator_winding_alarm"]
        )

        # 3. Fatti idraulici - Sistema di controllo del passo pale (Pitch System)
        facts["hydraulic_pressure_low"] = bool(row["hydraulic_pressure_bar"] < self.thresholds["hydraulic_min_safe"])
        facts["hydraulic_pressure_critical"] = bool(row["hydraulic_pressure_bar"] < (self.thresholds["hydraulic_min_safe"] - 15.0))

        # 4. Fatti cinematici ed elettrici
        facts["generator_overspeed"] = bool(row["generator_rpm"] >= self.thresholds["max_rpm_overspeed"])
        
        # Curva teorica di potenza approssimata: P_theor ~ 0.5 * rho * A * Cp * v^3
        theoretical_power = np.clip((row["wind_speed_ms"] ** 3) * 2.2, 0.0, 2000.0)
        actual_power = row["active_power_kw"]
        
        if theoretical_power > 150.0:
            efficiency_ratio = actual_power / theoretical_power
            facts["power_curve_underperformance"] = bool(efficiency_ratio < self.thresholds["min_power_curve_efficiency"])
        else:
            facts["power_curve_underperformance"] = False

        # 5. Fatti statistici (Z-Score su finestre mobili se presenti)
        zscore_gearbox = row.get("gearbox_oil_temp_c_zscore", 0.0)
        zscore_bearing = row.get("gearbox_bearing_temp_c_zscore", 0.0)
        facts["thermal_runaway_detected"] = bool(
            (zscore_gearbox > config.ZSCORE_THRESHOLD) or (zscore_bearing > config.ZSCORE_THRESHOLD)
        )

        return facts

    def infer_subsystem_faults(self, facts: Dict[str, bool]) -> List[str]:
        """
        Motore di inferenza in avanti (Forward Chaining) per diagnosticare
        guasti specifici a livello di sottosistema.
        """
        active_faults = []

        # R1: Degrado / Guasto Meccanico al Moltiplicatore (Gearbox Failure)
        # IF (Gearbox Oil Critical OR Bearing Temp Critical) AND Thermal Runaway THEN Gearbox Mechanical Failure
        if (facts["gearbox_oil_critical"] or facts["bearing_temp_critical"]) and facts["thermal_runaway_detected"]:
            active_faults.append("FAULT_GEARBOX_MECHANICAL_FAILURE")

        # R2: Surriscaldamento Olio Trasmissione
        # IF Gearbox Oil High AND NOT Gearbox Oil Critical THEN Gearbox Thermal Warning
        elif facts["gearbox_oil_high"]:
            active_faults.append("WARN_GEARBOX_OIL_OVERHEAT")

        # R3: Guasto Idraulico Sistema Pitch
        # IF Hydraulic Pressure Critical THEN Pitch Actuator Block
        if facts["hydraulic_pressure_critical"]:
            active_faults.append("FAULT_HYDRAULIC_PITCH_ACTUATOR_COLLAPSE")
        elif facts["hydraulic_pressure_low"]:
            active_faults.append("WARN_HYDRAULIC_PRESSURE_LOW")

        # R4: Guasto Avvolgimenti / Isolamento Generatore
        # IF Generator Winding Critical AND Generator Overspeed THEN Generator Thermal Breakdown
        if facts["generator_winding_critical"] and facts["generator_overspeed"]:
            active_faults.append("FAULT_GENERATOR_ELECTRICAL_BREAKDOWN")
        elif facts["generator_winding_critical"]:
            active_faults.append("WARN_GENERATOR_WINDING_OVERHEAT")

        # R5: Anomalia Aerodinamica Pale / Pitch Misalignment
        # IF Power Underperformance AND NOT Hydraulic Low THEN Pitch Angle Misalignment
        if facts["power_curve_underperformance"] and not facts["hydraulic_pressure_low"]:
            active_faults.append("WARN_AERODYNAMIC_PITCH_MISALIGNMENT")

        return active_faults

    def diagnose_health_state(self, facts: Dict[str, bool], active_faults: List[str]) -> Tuple[int, str]:
        """
        Inferenza dello stato di salute generale (HEALTHY, WARNING, CRITICAL).
        """
        critical_fault_keywords = ["FAULT_GEARBOX_MECHANICAL_FAILURE", "FAULT_HYDRAULIC_PITCH_ACTUATOR_COLLAPSE", "FAULT_GENERATOR_ELECTRICAL_BREAKDOWN"]
        
        has_critical_fault = any(f in critical_fault_keywords for f in active_faults)
        has_critical_telemetry = facts["gearbox_oil_critical"] or facts["bearing_temp_critical"] or facts["hydraulic_pressure_critical"]

        if has_critical_fault or has_critical_telemetry:
            return config.HEALTH_STATES["CRITICAL"], "CRITICAL"

        warning_keywords = ["WARN_GEARBOX_OIL_OVERHEAT", "WARN_HYDRAULIC_PRESSURE_LOW", "WARN_GENERATOR_WINDING_OVERHEAT", "WARN_AERODYNAMIC_PITCH_MISALIGNMENT"]
        has_warning_fault = any(f in warning_keywords for f in active_faults)

        if has_warning_fault or facts["gearbox_oil_high"] or facts["hydraulic_pressure_low"] or facts["thermal_runaway_detected"]:
            return config.HEALTH_STATES["WARNING"], "WARNING"

        return config.HEALTH_STATES["HEALTHY"], "HEALTHY"


class LogicEngine:
    """
    Interfaccia ad alto livello per eseguire l'inferenza logica
    su interi dataset o singoli campioni di telemetria.
    """

    def __init__(self):
        self.kb = WindTurbineKnowledgeBase()

    def evaluate_sample(self, row: pd.Series) -> Dict[str, Any]:
        """
        Valuta un singolo record temporale di una turbina.
        """
        facts = self.kb.evaluate_telemetry_facts(row)
        faults = self.kb.infer_subsystem_faults(facts)
        state_code, state_name = self.kb.diagnose_health_state(facts, faults)

        return {
            "turbine_id": row.get(config.ID_COL, "UNKNOWN"),
            "operating_hours": row.get(config.TIME_COL, 0),
            "inferred_state_code": state_code,
            "inferred_state_name": state_name,
            "active_faults": faults,
            "facts": facts
        }

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Esegue l'inferenza su un intero DataFrame di telemetria
        restituendo un DataFrame con le diagnosi logiche.
        """
        results = []
        for _, row in df.iterrows():
            diag = self.evaluate_sample(row)
            results.append({
                config.ID_COL: diag["turbine_id"],
                config.TIME_COL: diag["operating_hours"],
                "kb_predicted_state": diag["inferred_state_code"],
                "kb_state_name": diag["inferred_state_name"],
                "fault_count": len(diag["active_faults"]),
                "fault_details": ";".join(diag["active_faults"]) if diag["active_faults"] else "NONE"
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("--- Test Motore Logico (logic_engine.py) ---")
    import data_loader

    # Carica o genera i dati
    dataset = data_loader.load_dataset()
    
    # Esegui l'inferenza logica
    engine = LogicEngine()
    kb_results = engine.evaluate_dataframe(dataset)
    
    # Unione con i dati originali per confronto
    merged = dataset.merge(kb_results, on=[config.ID_COL, config.TIME_COL])
    
    print("\nDistribuzione Stati Inferiti dal Motore Logico:")
    print(merged["kb_state_name"].value_counts())
    
    print("\nEsempio di record critici individuati dalle regole:")
    critical_samples = merged[merged["kb_predicted_state"] == config.HEALTH_STATES["CRITICAL"]].head(3)
    print(critical_samples[[config.ID_COL, config.TIME_COL, "gearbox_oil_temp_c", "hydraulic_pressure_bar", "kb_state_name", "fault_details"]])