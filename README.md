# KARE — Knowledge-based Aircraft Risk & Engine Maintenance

Sistema di Ingegneria della Conoscenza per diagnosi del degrado e pianificazione della manutenzione predittiva su dataset NASA C-MAPSS.

## Struttura

```text
NASA C-MAPSS Dataset
↓
data_loader.py
↓
logic_engine.py        # Knowledge Base pyDatalog
↓
bayesian_learner.py    # Rete Bayesiana pgmpy
↓
maintenance_optimizer.py # CSP python-constraint
↓
main.py + evaluation scripts
```

## Dataset

Metti i file C-MAPSS in:

```text
data/CMAPSSData/
  train_FD001.txt
  test_FD001.txt
  train_FD002.txt
  ...
```

Oppure passa il percorso esplicito:

```bash
python main.py --subset FD001 --data-dir /percorso/CMAPSSData
```

## Comandi principali

Analisi completa:

```bash
python main.py --subset FD001 --run-analysis
```

Cross-validation rete bayesiana:

```bash
python main.py --subset FD001 --cross-validate --k 5
```

Valutazione KB:

```bash
python main.py --subset FD001 --evaluate-kb --k 5
```

Valutazione CSP:

```bash
python main.py --subset FD001 --evaluate-csp --k 5
```

Confronto con baseline supervisionate:

```bash
python main.py --subset FD001 --compare-models --k 5
```

Report di un motore specifico:

```bash
python main.py --subset FD001 --engine-id 42
```

Esperimenti completi:

```bash
python experiment_runner.py --subset FD001 --k 5 --json-out results_fd001.json
```

## Note metodologiche

- La cross-validation usa `GroupKFold` su `engine_id`, così i cicli dello stesso motore non finiscono contemporaneamente in train e test.
- La Knowledge Base non usa RUL né target reali: inferisce degrado e urgenza solo da stati sensoriali simbolici.
- La rete bayesiana usa anche gli output KB come evidenza probabilistica.
- Il CSP pianifica manutenzione assegnando giorno, slot, tecnico e intervento, rispettando deadline, budget e capacità.
