"""
bayesian_learner.py - Modulo di modellazione e inferenza probabilistica 
basato su Reti Bayesiane per la stima dello stato di salute di turbine eoliche.

Ruoli del modulo:
DAG Causale Strutturato: Modella le dipendenze fisiche (es. Vento -> Temperatura Olio -> Usura Cuscinetto -> Stato di Salute).

Discretizzazione Intelligente: Mappa le grandezze continue SCADA in stati di allarme qualitativi coerenti con le soglie fisiche di config.py.

Calcolo Probabilità e Failure Risk Score: Restituisce la distribuzione posteriore P(Stato | Sensori) e un indice di rischio composito continuo (failure_risk_score)
compreso tra 0 e 1.0.

"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import config

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


class WindTurbineBayesianLearner:
    """
    Rete Bayesiana per la prognostica di salute delle turbine eoliche.
    Struttura del DAG orientato causalmente:
      [Vento / Stress Ambientale] -> [Stress Meccanico / Termico]
      [Pressione Idraulica]       -> [Degrado Pitch]
      [Stress Meccanico] + [Degrado Pitch] -> [Health State]
    """

    def __init__(self):
        self.health_states = config.HEALTH_STATES
        self.model: Optional[Any] = None
        self.inference_engine: Optional[Any] = None
        self.is_fitted = False
        
        # Nomi delle variabili discretizzate nel DAG Bayesiano
        self.discrete_vars = [
            "wind_regime",           # Basso, Nominale, Alto/Tempesta
            "gearbox_thermal_state", # Normale, Alto, Critico
            "bearing_stress",        # Normale, Alto
            "hydraulic_status",      # Normale, Bassa Pressione
            "power_efficiency",      # Efficiente, Bassa Efficienza
            "health_state"           # 0=HEALTHY, 1=WARNING, 2=CRITICAL
        ]

    def discretize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Discretizza le variabili continue dei sensori SCADA in stati qualitativi finiti
        per consentire l'apprendimento delle tabelle di probabilità condizionata (CPD).
        """
        df_disc = pd.DataFrame(index=df.index)

        # 1. Regime del Vento
        # 0: Basso (<6 m/s), 1: Nominale (6-13 m/s), 2: Alto/Turbolento (>13 m/s)
        df_disc["wind_regime"] = pd.cut(
            df["wind_speed_ms"],
            bins=[-np.inf, 6.0, 13.0, np.inf],
            labels=[0, 1, 2]
        ).astype(int)

        # 2. Stato Termico Moltiplicatore
        # 0: Safe (<80°C), 1: Warning (80-95°C), 2: Critical (>=95°C)
        df_disc["gearbox_thermal_state"] = pd.cut(
            df["gearbox_oil_temp_c"],
            bins=[-np.inf, config.PHYSICAL_THRESHOLDS["gearbox_oil_max_safe"], config.PHYSICAL_THRESHOLDS["gearbox_oil_alarm"], np.inf],
            labels=[0, 1, 2]
        ).astype(int)

        # 3. Stress Cuscinetto (Bearing)
        # 0: Normale (<90°C), 1: Alto (>=90°C)
        df_disc["bearing_stress"] = (df["gearbox_bearing_temp_c"] >= config.PHYSICAL_THRESHOLDS["bearing_temp_alarm"]).astype(int)

        # 4. Stato Circuito Idraulico Pitch
        # 0: Normale (>=140 bar), 1: Bassa Pressione (<140 bar)
        df_disc["hydraulic_status"] = (df["hydraulic_pressure_bar"] < config.PHYSICAL_THRESHOLDS["hydraulic_min_safe"]).astype(int)

        # 5. Efficienza della Curva di Potenza
        theoretical_power = np.clip((df["wind_speed_ms"] ** 3) * 2.2, 1.0, 2000.0)
        eff_ratio = df["active_power_kw"] / theoretical_power
        df_disc["power_efficiency"] = (eff_ratio < config.PHYSICAL_THRESHOLDS["min_power_curve_efficiency"]).astype(int)

        # Target: Health State
        if "health_state" in df.columns:
            df_disc["health_state"] = df["health_state"].astype(int)

        return df_disc

    def define_network_structure(self) -> List[Tuple[str, str]]:
        """
        Definisce la topologia del Grafo Aciclico Diretto (DAG)
        basata su causalità fisica tra condizioni operative e usura.
        """
        edges = [
            ("wind_regime", "gearbox_thermal_state"),
            ("wind_regime", "power_efficiency"),
            ("gearbox_thermal_state", "bearing_stress"),
            ("gearbox_thermal_state", "health_state"),
            ("bearing_stress", "health_state"),
            ("hydraulic_status", "health_state"),
            ("power_efficiency", "health_state")
        ]
        return edges

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Addestra i parametri della Rete Bayesiana stimando le CPD
        tramite Maximum Likelihood / Bayesian Estimation con smoothing di Laplace.
        """
        df_disc = self.discretize_dataset(df_train)
        edges = self.define_network_structure()

        if PGMPY_AVAILABLE:
            self.model = BayesianNetwork(edges)
            # Stima Bayesiana con prior Dirichlet per evitare probabilità zero
            self.model.fit(df_disc, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
            self.inference_engine = VariableElimination(self.model)
        else:
            print("[WARN] pgmpy non installata. Utilizzo fallback euristico su frequenze congiunte.")
            self._fit_fallback(df_disc)

        self.is_fitted = True

    def predict_health_probability(self, evidence: Dict[str, int]) -> Dict[str, float]:
        """
        Esegue l'inferenza probabilistica (Exact Variable Elimination)
        calcolando P(health_state | Evidenza Sensori).
        """
        if not self.is_fitted:
            raise RuntimeError("Il modello non è stato addestrato. Chiama fit() prima dell'inferenza.")

        # Filtra l'evidenza solo sulle variabili del DAG escludendo la target
        valid_evidence = {k: v for k, v in evidence.items() if k in self.discrete_vars and k != "health_state"}

        if PGMPY_AVAILABLE and self.inference_engine is not None:
            query_result = self.inference_engine.query(
                variables=["health_state"],
                evidence=valid_evidence,
                show_progress=False
            )
            probs = query_result.values
            return {
                "P(HEALTHY)": float(probs[config.HEALTH_STATES["HEALTHY"]]),
                "P(WARNING)": float(probs[config.HEALTH_STATES["WARNING"]]),
                "P(CRITICAL)": float(probs[config.HEALTH_STATES["CRITICAL"]])
            }
        else:
            return self._predict_fallback(valid_evidence)

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola le probabilità di degrado e predice lo stato a massima verosimiglianza (MAP)
        per ogni istante temporale.
        """
        df_disc = self.discretize_dataset(df)
        results = []

        for idx, row in df_disc.iterrows():
            evidence = row.to_dict()
            evidence.pop("health_state", None)
            probs = self.predict_health_probability(evidence)
            
            # Criterio MAP (Maximum A Posteriori)
            p_healthy = probs["P(HEALTHY)"]
            p_warning = probs["P(WARNING)"]
            p_critical = probs["P(CRITICAL)"]
            
            predicted_state = int(np.argmax([p_healthy, p_warning, p_critical]))

            results.append({
                "bayes_pred_state": predicted_state,
                "prob_healthy": round(p_healthy, 4),
                "prob_warning": round(p_warning, 4),
                "prob_critical": round(p_critical, 4),
                "failure_risk_score": round(p_warning * 0.4 + p_critical * 1.0, 4)
            })

        res_df = pd.DataFrame(results, index=df.index)
        return pd.concat([df[[config.ID_COL, config.TIME_COL]], res_df], axis=1)

    # --- Metodi di Fallback se pgmpy non è installata nell'ambiente ---
    def _fit_fallback(self, df_disc: pd.DataFrame) -> None:
        self.fallback_cpt = df_disc.groupby(["gearbox_thermal_state", "bearing_stress", "hydraulic_status", "power_efficiency"])["health_state"].value_counts(normalize=True).unstack(fill_value=0.0)

    def _predict_fallback(self, evidence: Dict[str, int]) -> Dict[str, float]:
        key = (
            evidence.get("gearbox_thermal_state", 0),
            evidence.get("bearing_stress", 0),
            evidence.get("hydraulic_status", 0),
            evidence.get("power_efficiency", 0)
        )
        if hasattr(self, "fallback_cpt") and key in self.fallback_cpt.index:
            row = self.fallback_cpt.loc[key]
            return {
                "P(HEALTHY)": float(row.get(0, 0.33)),
                "P(WARNING)": float(row.get(1, 0.33)),
                "P(CRITICAL)": float(row.get(2, 0.34))
            }
        # Prior di default
        return {"P(HEALTHY)": 0.60, "P(WARNING)": 0.30, "P(CRITICAL)": 0.10}


if __name__ == "__main__":
    print("--- Test Rete Bayesiana (bayesian_learner.py) ---")
    import data_loader

    # Carica dataset
    data = data_loader.load_dataset()
    
    # Inizializza e addestra
    learner = WindTurbineBayesianLearner()
    learner.fit(data)
    print("[OK] Rete Bayesiana addestrata con successo.")

    # Inferenza su tutto il dataset
    bayes_predictions = learner.evaluate_dataframe(data)
    print("\nAnteprima predizioni probabilistiche e Rischio Guasto:")
    print(bayes_predictions.head(8))