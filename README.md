# ⚡ YIELD HUNTER — Intelligent Yield & Opportunity Analysis Platform

**Enterprise Semiconductor Process Analytics, Precision Root-Cause Diagnostics & Yield Recovery Engine**  
*IBM Bob AI Innovation Hackathon — CHARUSAT*

---

## 🎯 Executive Summary & Problem Statement

In semiconductor fabrication facilities, wafer lot yield (the percentage of functional dies per wafer) is the single most critical manufacturing KPI. Small excursions in process chamber conditions directly lead to wafer defect clusters and substantial yield losses.

When yield drops, process integration and yield engineers must quickly:
1. **Identify and isolate root causes** from multivariate chamber sensors and defect inspection maps.
2. **Quantify yield recovery opportunities** across affected lots.
3. **Obtain prescriptive corrective actions** (furnace calibration, vacuum regulator repair, RF tuning).
4. **Pre-screen upcoming lots** before physical dispatch to prevent scrap.

**Yield Hunter** transforms this manual, error-prone workflow into an automated, interactive, and transparent SaaS-grade analytics platform.

---

## 🔄 End-to-End User Flow

```
OPEN WEBSITE
     ↓
 DATA INPUT (Upload CSV / Try Demo Dataset / Download Template)
     ↓
 CSV VALIDATION & INTEGRITY AUDIT
     ↓
 DATA PREVIEW (First 10 rows, rows/cols count)
     ↓
 COLUMN MAPPING (Intelligent auto-match + manual override)
     ↓
 🚀 RUN YIELD ANALYSIS (Manual on-demand trigger with spinner)
     ↓
 PROFESSIONAL RESULTS DASHBOARD (KPIs, Plotly Analytics, Excursions)
     ↓
 📥 DOWNLOAD RESULTS CSV
```

---

## 🛠️ Required vs. Optional Columns

Yield Hunter supports flexible dataset schemas via automated and manual **Column Mapping**. The core algorithms require:

| Standard Field | Requirement | Description | Auto-Match Heuristics |
|---|---|---|---|
| `Yield` | **Required** | Actual lot yield percentage (0–100%) | `yield`, `yield%`, `yield_pct`, `recovery`, `output_yield` |
| `Defects` | **Required** | Defect count / particle density | `defects`, `defect`, `defect_count`, `particles`, `errors` |
| `Temperature` | **Required** | Process chamber temperature (°C) | `temperature`, `temp`, `temp_c`, `temp_deg`, `chamber_temp` |
| `Pressure` | **Required** | Chamber pressure (torr / bar) | `pressure`, `press`, `chamber_pressure`, `pressure_torr` |
| `Power` | **Required** | RF / heater power (Watts) | `power`, `rf_power`, `watts`, `heater_power` |
| `GasFlow` | Optional | Mass flow controller rate (sccm) | `gasflow`, `gas_flow`, `flow`, `sccm`, `gas_sccm` |
| `Lot` | Optional | Lot / batch identifier (auto-generated if omitted) | `lot`, `lot_id`, `wafer_lot`, `batch`, `run_id` |

---

## ✨ Key Features & Capabilities

- **Real CSV Uploader & Ingestion**: Direct pandas-backed file parsing with strict error handling, schema auditing, and format validation.
- **Smart Column Mapping**: Automatically detects and maps your CSV column names to engine requirements with manual dropdown confirmation.
- **Data Quality Audit**: Immediate visibility into row count, clean valid rows, duplicate rows, missing nulls, and non-numeric value flags.
- **On-Demand Analysis**: Heavy model training and analytics run only when clicking **🚀 Run Yield Analysis**, preventing unnecessary recalculations.
- **Modern Dark Navy Analytics Theme**: High-contrast, responsive SaaS styling with rounded cards, subtle glows, and clean typography.
- **Real Calculated KPI Metrics**: Audits total lots, average yield, median yield, best lot yield, low-yield excursion count, and opportunity gap.
- **Interactive Plotly Visualizations**:
  - Lot yield distribution with excursion cutoff and mean benchmarks.
  - Root-cause attribution via Random Forest feature importance.
  - Parameter-vs-yield interactive scatter explorer with defect heatmaps and OLS trendlines.
  - Parameter correlation matrix heatmap.
  - Run-sequence lot yield trendline.
- **Actionable Opportunity Sizing**: Identifies underperforming lots, quantifies individual yield recovery gaps, diagnoses primary excursion factors, and prescribes fab-validated maintenance actions.
- **Results Export**: One-click download of the complete analyzed results CSV.
- **Pre-Dispatch Future Lot Predictor**: Enter planned parameters for an upcoming lot to forecast yield and risk tier (🟢 LOW / 🟡 MEDIUM / 🔴 HIGH) with z-score diagnostics.

---

## 💻 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Krishu50/bob-ai-hackathon--Yield-Hunters-.git
cd bob-ai-hackathon--Yield-Hunters-

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run unit tests
python test_engine.py

# 4. Launch the Streamlit application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Run the test suite at any time:
```bash
python test_engine.py
```
All 11 unit tests verify schema validation, summary metrics, random forest importance, risk prediction, lot opportunity sizing, and template generation.

---

## ☁️ Streamlit Cloud Deployment

The repository is fully optimized for continuous deployment on **Streamlit Cloud**:
- Dependencies in `requirements.txt` (`streamlit>=1.32`, `pandas>=2.0`, `numpy>=1.24`, `scikit-learn>=1.3`, `plotly>=5.18`, `statsmodels>=0.14`).
- Relative file paths and session state caching.
- No special native dependencies or system packages required.

