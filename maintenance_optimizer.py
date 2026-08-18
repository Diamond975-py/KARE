"""
maintenance_optimizer.py - Ottimizzatore CSP (Constraint Satisfaction Problem)

Questo modulo implementa il terzo pilastro fondamentale del sistema: la Risoluzione di Problemi di Soddisfacimento di Vincoli (CSP - Constraint Satisfaction Problem) 
e l'Ottimizzazione Combinatoria per pianificare gli interventi di manutenzione preventiva/correttiva sulle turbine eoliche

Ruoli del modulo:

Modellazione CSP a Vincoli Rigidi (Hard Constraints): Nessuna squadra opera su due aerogeneratori nello stesso giorno e nessun tecnico sale in navicella con vento oltre i 12 m/s 

Prioritizzazione Dinamica: Le turbine con guasto logico imminente (is_critical) hanno un deadline inderogabile.

Integrazione Multi-Algoritmica: Riceve le diagnosi del Motore Logico (logic_engine.py) e i punteggi di rischio della Rete Bayesiana (bayesian_learner.py) per produrre la tabella operativa finale.

"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import config

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


class WindFarmMaintenanceOptimizer:
    """
    Ottimizzatore basato su Constraint Programming (CP-SAT) per pianificare
    gli interventi di manutenzione sulle turbine eoliche.
    
    Variabili decisionali:
      - X[t, d, c] ∈ {0, 1}: Turbina t manutenuta nel giorno d dalla squadra c.
    
    Vincoli (Hard Constraints):
      1. Ogni turbina con allarme/rischio elevato viene riparata al più una volta.
      2. Le turbine con RUL critica devono essere riparate prima del deadline Day_max.
      3. Capacità giornaliera: ogni squadra non può riparare più di 1 turbina al giorno (interventi lunghi).
      4. Vincolo meteo (Wind Safety): nessun intervento se la velocità del vento prevista supera la soglia di sicurezza.
      
    Obiettivo (Objective Function):
      Minimizzare: Costi di Intervento + Perdite Economiche da Fermo + Penalità Rischio Residuo.
    """

    def __init__(self, planning_days: int = 14, max_crews: int = 2):
        self.planning_days = planning_days
        self.max_crews = max_crews
        self.cfg = config.CSP_CONFIG
        self.weather_wind_limit = self.cfg["weather_wind_speed_limit_ms"]

    def generate_weather_forecast(self, seed: int = 42) -> np.ndarray:
        """
        Genera le previsioni meteo giornaliere di velocità del vento per l'orizzonte temporale.
        """
        np.random.seed(seed)
        # Giorni con vento medio tra 6 e 16 m/s
        return np.random.uniform(5.0, 16.5, size=self.planning_days)

    def extract_candidates_from_predictions(self, df_predictions: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Estrae l'ultimo stato noto di ogni turbina identificando quelle che
        richiedono intervento (Warning, Critical o alto Failure Risk Score).
        """
        # Considera l'ultimo record temporale di ciascuna turbina
        latest = df_predictions.sort_values(config.TIME_COL).groupby(config.ID_COL).last().reset_index()
        
        candidates = []
        for _, row in latest.iterrows():
            t_id = row[config.ID_COL]
            rul = row.get("RUL", 150.0)
            risk_score = row.get("failure_risk_score", 0.1)
            kb_state = row.get("kb_predicted_state", config.HEALTH_STATES["HEALTHY"])
            
            # Turbina candidata se in allarme KB o con rischio Bayesiano significativo
            if kb_state != config.HEALTH_STATES["HEALTHY"] or risk_score >= 0.35 or rul <= config.WARNING_RUL_THRESHOLD:
                # Calcola il deadline in giorni (convertendo RUL in ore / 24)
                days_until_failure = max(1, int(rul // 24))
                deadline_day = min(self.planning_days, days_until_failure)
                
                is_critical = (kb_state == config.HEALTH_STATES["CRITICAL"]) or (rul <= config.CRITICAL_RUL_THRESHOLD)
                
                candidates.append({
                    "turbine_id": t_id,
                    "rul_hours": rul,
                    "risk_score": risk_score,
                    "is_critical": is_critical,
                    "deadline_day": deadline_day,
                    "intervention_cost": self.cfg["cost_corrective_intervention"] if is_critical else self.cfg["cost_preventive_intervention"],
                    "daily_downtime_loss": self.cfg["cost_power_loss_per_hour"] * 8  # 8 ore di fermo per intervento
                })
                
        return candidates

    def solve_ortools(self, candidates: List[Dict[str, Any]], weather_forecast: np.ndarray) -> Dict[str, Any]:
        """
        Risolutore CSP esatto tramite Google OR-Tools CP-SAT.
        """
        model = cp_model.CpModel()
        
        num_turbines = len(candidates)
        if num_turbines == 0:
            return {"status": "NO_CANDIDATES", "schedule": [], "total_cost": 0}

        # Indici
        T = range(num_turbines)
        D = range(self.planning_days)
        C = range(self.max_crews)

        # Variabili decisionali binarie: x[t, d, c] == 1 se la turbina t è manutenuta nel giorno d dalla squadra c
        x = {}
        for t in T:
            for d in D:
                for c in C:
                    x[t, d, c] = model.NewBoolVar(f"x_{t}_{d}_{c}")

        # --- VINCOLI (CONSTRAINTS) ---
        
        # 1. Al massimo un intervento per turbina nell'orizzonte temporale
        for t in T:
            model.Add(sum(x[t, d, c] for d in D for c in C) <= 1)

        # 2. Le turbine CRITICHE devono essere obbligatoriamente riparate entro il deadline
        for t in T:
            if candidates[t]["is_critical"]:
                deadline = candidates[t]["deadline_day"]
                # Deve essere riparata prima o entro il deadline
                model.Add(sum(x[t, d, c] for d in range(deadline) for c in C) == 1)

        # 3. Capacità delle squadre: al massimo 1 turbina per squadra al giorno
        for d in D:
            for c in C:
                model.Add(sum(x[t, d, c] for t in T) <= 1)

        # 4. Vincolo meteo: vietata la salita in quota se il vento supera il limite di sicurezza
        for d in D:
            if weather_forecast[d] > self.weather_wind_limit:
                for t in T:
                    for c in C:
                        model.Add(x[t, d, c] == 0)

        # --- FUNZIONE OBIETTIVO (MINIMIZZAZIONE COSTI) ---
        objective_terms = []
        
        for t in T:
            t_data = candidates[t]
            cost_base = t_data["intervention_cost"]
            
            # Se la turbina viene riparata nel giorno d, paghiamo il costo base + le perdite di fermo
            for d in D:
                for c in C:
                    # Più tardi ripariamo una turbina critica, maggiore è il rischio
                    delay_penalty = int(d * t_data["risk_score"] * 300)
                    cost_schedule = cost_base + delay_penalty
                    objective_terms.append(x[t, d, c] * cost_schedule)
            
            # Penalità severa se una turbina non critica non viene programmata affatto
            is_serviced = sum(x[t, d, c] for d in D for c in C)
            unserviced_risk_penalty = int(t_data["risk_score"] * 8000)
            objective_terms.append((1 - is_serviced) * unserviced_risk_penalty)

        model.Minimize(sum(objective_terms))

        # Risoluzione
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        schedule = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for t in T:
                for d in D:
                    for c in C:
                        if solver.Value(x[t, d, c]) == 1:
                            schedule.append({
                                "day": d + 1,
                                "crew_id": f"Crew_{c + 1}",
                                "turbine_id": candidates[t]["turbine_id"],
                                "is_critical": candidates[t]["is_critical"],
                                "wind_forecast_ms": round(float(weather_forecast[d]), 2),
                                "estimated_rul_hours": round(float(candidates[t]["rul_hours"]), 1),
                                "risk_score": round(float(candidates[t]["risk_score"]), 3)
                            })
            schedule = sorted(schedule, key=lambda item: (item["day"], item["crew_id"]))
            return {
                "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                "schedule": schedule,
                "total_cost": solver.ObjectiveValue()
            }
        else:
            return {"status": "INFEASIBLE", "schedule": [], "total_cost": None}

    def solve_greedy_fallback(self, candidates: List[Dict[str, Any]], weather_forecast: np.ndarray) -> Dict[str, Any]:
        """
        Risolutore euristico greedy rapido (fallback nel caso OR-Tools non sia presente).
        """
        sorted_candidates = sorted(candidates, key=lambda x: (not x["is_critical"], x["deadline_day"], -x["risk_score"]))
        schedule = []
        
        # Mappa occupazione: crew_schedule[crew_idx][day_idx] = True/False
        crew_busy = {c: [False] * self.planning_days for c in range(self.max_crews)}
        
        for cand in sorted_candidates:
            assigned = False
            for d in range(min(cand["deadline_day"], self.planning_days)):
                if weather_forecast[d] <= self.weather_wind_limit:
                    for c in range(self.max_crews):
                        if not crew_busy[c][d]:
                            crew_busy[c][d] = True
                            schedule.append({
                                "day": d + 1,
                                "crew_id": f"Crew_{c + 1}",
                                "turbine_id": cand["turbine_id"],
                                "is_critical": cand["is_critical"],
                                "wind_forecast_ms": round(float(weather_forecast[d]), 2),
                                "estimated_rul_hours": round(float(cand["rul_hours"]), 1),
                                "risk_score": round(float(cand["risk_score"]), 3)
                            })
                            assigned = True
                            break
                    if assigned:
                        break
                        
        return {"status": "HEURISTIC_SOLVED", "schedule": schedule, "total_cost": 0}

    def plan_maintenance(self, df_predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline completa: estrazione candidati, generazione meteo e risoluzione vincoli.
        """
        candidates = self.extract_candidates_from_predictions(df_predictions)
        weather = self.generate_weather_forecast()
        
        if ORTOOLS_AVAILABLE:
            result = self.solve_ortools(candidates, weather)
            if result["status"] == "INFEASIBLE":
                print("[WARN] CSP non fattibile con vincoli rigidi. Fallback su euristica rilassata.")
                result = self.solve_greedy_fallback(candidates, weather)
        else:
            print("[INFO] OR-Tools non installata. Esecuzione pianificatore euristico.")
            result = self.solve_greedy_fallback(candidates, weather)
            
        result["weather_forecast_ms"] = [round(float(w), 2) for w in weather]
        return result


if __name__ == "__main__":
    print("--- Test Pianificatore CSP Manutenzione (maintenance_optimizer.py) ---")
    import data_loader
    import logic_engine
    import bayesian_learner

    # 1. Pipeline dati ed inferenze
    df = data_loader.load_dataset()
    
    le = logic_engine.LogicEngine()
    df_logic = le.evaluate_dataframe(df)
    
    bl = bayesian_learner.WindTurbineBayesianLearner()
    bl.fit(df)
    df_bayes = bl.evaluate_dataframe(df)
    
    # 2. Merge previsioni
    df_combined = df.merge(df_logic, on=[config.ID_COL, config.TIME_COL])
    df_combined["failure_risk_score"] = df_bayes["failure_risk_score"]
    
    # 3. Ottimizzazione CSP
    optimizer = WindFarmMaintenanceOptimizer(planning_days=10, max_crews=2)
    plan = optimizer.plan_maintenance(df_combined)
    
    print(f"\nStato Risoluzione CSP: {plan['status']}")
    print(f"Previsioni Vento Giornaliere (m/s, max consentito {config.CSP_CONFIG['weather_wind_speed_limit_ms']} m/s):")
    print(plan["weather_forecast_ms"])
    
    print("\nPiano Manutentivo Ottimizzato Generato:")
    df_plan = pd.DataFrame(plan["schedule"])
    if not df_plan.empty:
        print(df_plan.to_string(index=False))
    else:
        print("Nessuna turbina richiede manutenzione immediata o condizioni meteo avverse.")
