# 🏭 Wafer Yield Root-Cause & Risk Prediction Dashboard

**IBM Bob AI Innovation Hackathon — CHARUSAT**

## Problem Statement

Semiconductor fabs run many wafer lots, and yield (the % of good chips produced)
fluctuates lot to lot. When yield drops, process engineers need to quickly:
1. Understand *why* yield dropped
2. Know which process parameters are most likely responsible
3. Get concrete next steps to investigate
4. Flag upcoming lots that are at risk **before** they run, based on planned
   process parameters

Doing this manually from raw process logs and defect reports is slow and
inconsistent. This project is a decision-support prototype that automates it.

## Proposed Solution

A Streamlit dashboard that takes historical wafer-lot data (Lot ID, Temperature,
Pressure, Power, Gas Flow, Defect count, Yield) and:

1. **Analyzes** the relationship between process parameters, defects, and yield
   (scatter plots, correlation matrix, yield trend)
2. **Ranks contributing factors** using a Random Forest model's feature
   importances — trained to predict yield from process parameters
3. **Generates rule-based recommendations** — flags any parameter more than
   1.5 standard deviations from the "healthy lot" median and suggests a
   concrete investigation action (check sensor calibration, inspect for leaks,
   review PM records, etc.)
4. **Predicts risk for a new/upcoming lot** — engineer enters planned
   Temperature/Pressure/Power/Defects/Gas Flow, and the trained model predicts
   expected yield and classifies the lot as 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH risk,
   with the specific abnormal parameters called out

**Important framing:** the "root cause ranking" is a *statistical association*
ranking from model feature importance, not a proven causal analysis — this is
stated explicitly in the UI to stay scientifically honest for a fab context.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data handling | Pandas, NumPy |
| ML | Scikit-learn (RandomForestRegressor) |
| Visualization | Plotly Express |
| Dashboard/UI | Streamlit |

## Project Structure

```
wafer-yield-app/
├── app.py              # Streamlit UI (4 pages)
├── engine.py           # Core analysis/ML logic (no UI code — testable)
├── generate_data.py    # Creates the synthetic demo dataset
├── wafer_data.csv      # Demo dataset (500 lots)
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone the repo and cd into the project folder
git clone <your-repo-url>
cd wafer-yield-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the demo dataset
python generate_data.py

# 4. Launch the dashboard
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Using Your Own Data

Upload a CSV via the sidebar with (at minimum) these columns:

| Column | Description |
|---|---|
| `Lot` | Lot identifier (optional — auto-generated if missing) |
| `Temperature` | Process temperature |
| `Pressure` | Process pressure |
| `Power` | Process power |
| `Defects` | Defect count |
| `Yield` | Yield % |
| `GasFlow` | Gas flow rate (optional) |

The app validates columns, coerces numeric types, and drops invalid rows with
a warning shown in the sidebar.

## Dashboard Pages

1. **🏠 Dashboard** — total lots, average yield, low-yield lot count, yield
   distribution histogram
2. **📊 Yield & Defect Analysis** — yield trend across lots, parameter-vs-yield
   scatter plots with trendlines, correlation heatmap
3. **🎯 Root Cause** — ranked contributing factors bar chart + list of
   low-yield lots for investigation
4. **⚠️ Future Lot Prediction** — input form for a planned lot's parameters →
   predicted yield, risk level, flagged abnormal parameters, and recommended
   corrective actions

## What This Prototype Does NOT Do

This is a decision-support software prototype, not:
- An actual semiconductor fab simulation or physical sensor system
- A proven causal-inference engine (ranking is association-based, clearly
  labeled as such)
- A production-grade MLOps pipeline — model is retrained in-session for
  demo purposes

## Future Improvements

- SHAP values for per-lot explainability instead of global feature importance
- Time-series / drift detection (e.g. CUSUM) for early excursion alerts
- Multi-model ensemble with confidence intervals on predicted yield
- Integration with real fab MES/SPC systems via API
