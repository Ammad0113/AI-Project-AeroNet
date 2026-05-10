# AeroNet Lite

**Autonomous Drone Delivery Simulation**  
BS Data Science — AI Semester Project, SP2026 · FAST-NUCES Rawalpindi

---

## Overview

AeroNet Lite models a city as a 10×10 grid and simulates an autonomous drone delivery system across five integrated AI modules. The system validates city layout using constraint satisfaction, selects an optimal drone fleet with a genetic algorithm, plans routes using A* search, reacts to real-time disruptions, and uses machine learning for demand forecasting and anomaly detection.

---

## AI Techniques

| Module | Technique | Details |
|---|---|---|
| Layout Validator | Constraint Satisfaction Problem | 4 rules (R1–R4), neighbor and distance checking |
| Fleet Selector | Genetic Algorithm | Population 60, generations 80, mutation rate 0.2 |
| Fleet Selector | Brute Force | Exhaustive search across all light/heavy combinations |
| Path Planner | A* Search | Manhattan heuristic, 4-direction grid movement |
| Disruption Handler | Online Replanning | No-fly activation + A* reroute from current position |
| Demand Forecasting | Random Forest Regression | 100 estimators, max depth 10, Bike Sharing dataset |
| Anomaly Detection | Random Forest Classifier | 100 estimators, max depth 10, ALFA UAV dataset |

---

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| Bike Sharing Demand | Demand forecasting (regression) | [Kaggle](https://www.kaggle.com/c/bike-sharing-demand) |
| US City Population Densities | Grid population mapping | [Kaggle](https://www.kaggle.com/datasets/mmcgurr/us-city-population-densities) |
| ALFA UAV Telemetry | Anomaly detection (classification) | [CMU KiltHub](https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963) |

---

## Project Structure

```
aeronet_lite/
├── data/
│   ├── raw/
│   │   ├── bike-sharing-demand.zip
│   │   ├── archive (4).zip
│   │   └── processed.zip
│   └── processed/
│       ├── demand_model.pkl
│       ├── demand_scaler.pkl
│       ├── anomaly_model.pkl
│       ├── anomaly_scaler.pkl
│       └── anomaly_encoder.pkl
├── src/
│   ├── grid_model.py          # Core grid data structures
│   ├── layout_validator.py    # CSP constraint checker
│   ├── fleet_selector.py      # GA + brute force fleet selection
│   ├── astar_planner.py       # A* path planning
│   ├── delivery_simulator.py  # 20-step simulation engine
│   ├── ml_pipeline.py         # ML model loading and prediction
│   ├── visualization.py       # Matplotlib dashboard
│   ├── dashboard.py           # Streamlit web interface
│   └── load_datasets.py       # Dataset loading utilities
├── notebooks/
│   ├── demand_forecasting.ipynb
│   └── anomaly_classifier.ipynb
├── main.py                    # Console entry point
├── requirements.txt
└── README.md
```

---

## Installation

**1. Navigate to the project directory**
```bash
cd C:\Users\LOQ\Desktop\aeronet_lite
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Place datasets in `data/raw/`**

The following files must be present before training:
- `bike-sharing-demand.zip`
- `archive (4).zip`
- `processed.zip`

**4. Train ML models (first time only)**
```bash
cd notebooks
jupyter notebook demand_forecasting.ipynb   # run all cells
jupyter notebook anomaly_classifier.ipynb   # run all cells
cd ..
```

---

## Running the Project

### Option 1 — Console menu
```bash
python main.py
```

| Option | Action |
|---|---|
| 1 | Full 20-step simulation |
| 2 | Layout validation only (CSP) |
| 3 | Fleet selection only (GA + brute force) |
| 4 | A* path planner demo |
| 5 | ML pipeline demo |
| 6 | Visualizations only |
| 7 | Run everything (complete system test) |
| 8 | Exit |

### Option 2 — Streamlit dashboard
```bash
streamlit run src/dashboard.py
```
Then open `http://localhost:8501` in your browser.

### Option 3 — Visualization only
```bash
python src/visualization.py
```

---

## Testing Individual Modules

```bash
python src/grid_model.py          # Grid size, hub locations, cell properties
python src/layout_validator.py    # R1–R4 pass/fail with suggestions
python src/fleet_selector.py      # Fleet selection under budget
python src/astar_planner.py       # Path with cost and length
python src/ml_pipeline.py         # Demand predictions + anomaly labels
python src/visualization.py       # Master dashboard + 3D demand map
python src/load_datasets.py       # Dataset loading check
python src/delivery_simulator.py  # 20-step simulation event log
```

---

## Expected ML Output

**Demand Forecasting**
- MAE: 40–50
- R² Score: 0.85–0.90

**Anomaly Detection**
- Accuracy: 40–45%
- Classes: `failure`, `traj`, `truth`

---

## Simulation Output Example

```
======================================================================
AERONET LITE - DRONE DELIVERY SIMULATION
======================================================================

[Step 0]  Layout validation complete
[Step 0]  R1 PASSED — No industrial cells adjacent to schools or hospitals
[Step 0]  R2 FAILED — 1 violation found
[Step 0]  R3 PASSED — All hubs have charging pads within 2 cells
[Step 0]  R4 PASSED — At least one hospital has medical pickup within 1 cell

[Step 0]  Fleet selected: 1 Light + 5 Heavy = 6 total drones
[Step 0]  Total cost: $10,000  |  Fitness score: 1.2000

[Step 2]  Delivery 1: MEDICAL  (2,2) → (0,0)  1.1 kg  Priority 1
[Step 2]  Delivery 2: STANDARD (9,9) → (1,2)  0.6 kg  Priority 2
[Step 2]  Delivery 3: MEDICAL  (2,7) → (2,0)  1.3 kg  Priority 1
[Step 2]  Delivery 4: STANDARD (9,9) → (2,4)  0.6 kg  Priority 2

[Step 2]  Delivery 1 → Drone D001  (route: 9 cells)
[Step 2]  Delivery 3 → Drone D002  (route: 19 cells)
[Step 2]  Delivery 2 → Drone D003  (route: 37 cells)
[Step 2]  Delivery 4 → Drone D004  (route: 37 cells)

[Step 5]  Drone D001 completed Delivery 1

[Step 9]  DISRUPTION — No-fly cell activated at (4,7)
[Step 9]  Affected drones: D003, D004
[Step 9]  Drone D003 rerouted successfully
[Step 9]  Drone D004 rerouted successfully
[Step 9]  Drone D002 completed Delivery 3

[Step 14] Delivery 5: MEDICAL  (2,7) → (1,0)  1.0 kg  Priority 1
[Step 14] Delivery 6: STANDARD (5,5) → (2,1)  0.8 kg  Priority 2

[Step 18] Battery drain injected for D001
[Step 18] ANOMALY DETECTED — route_deviation  (confidence: 54.4%)

[Step 20] SIMULATION COMPLETE
[Step 20] Completed: 6  |  Failed: 0  |  Success rate: 100%
```

---

## Troubleshooting

**ModuleNotFoundError**  
Always run from the project root, not from inside `src/`:
```bash
cd C:\Users\LOQ\Desktop\aeronet_lite
python main.py
```

**Demand or anomaly model not found**  
Run the corresponding notebook first to train and save the `.pkl` files to `data/processed/`.

**Population density dataset not found**  
Ensure `data/raw/archive (4).zip` exists. The loader extracts it automatically.

**Streamlit command not recognised**  
```bash
python -m streamlit run src/dashboard.py
```

**Empty visualization dashboard**  
The visualizer generates its own sample fleet and deliveries. Wait for full rendering before interacting.

**KeyError on ZoneType**  
Use `.value` when comparing zone types. This is already handled in the current codebase.

---

## Submission Checklist

| Requirement | Status |
|---|---|
| Working Python project with clear folder structure | ✅ |
| 10×10 grid visualization | ✅ |
| CSP layout validator with failed rule reporting | ✅ |
| Fleet selection result under budget | ✅ |
| A* route planner | ✅ |
| Rerouting after disruption | ✅ |
| Regression model with MAE/R² | ✅ |
| Classifier with confusion matrix | ✅ |
| 20-step simulation event log | ✅ |
| README.md | ✅ |
| requirements.txt | ✅ |

---

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
streamlit>=1.20.0
plotly>=5.10.0
joblib>=1.1.0
```

---

## Acknowledgments

- Kaggle — Bike Sharing Demand dataset
- Kaggle — US City Population Densities dataset  
- CMU KiltHub — ALFA UAV Fault Detection dataset
- FAST-NUCES Rawalpindi — project guidance