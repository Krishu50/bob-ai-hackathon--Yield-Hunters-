"""
app.py
YIELD HUNTER — Intelligent Yield & Opportunity Analysis Platform
Wafer Fab Process Root-Cause, Yield Optimization & Risk Prediction Dashboard
"""

from __future__ import annotations

import io
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import (
    load_and_validate,
    compute_summary,
    compute_feature_importance,
    train_risk_model,
    predict_new_lot,
    get_feature_columns,
    compute_lot_opportunities,
    generate_template_csv,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Yield Hunter | Intelligent Yield & Opportunity Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Navy Professional Analytics Theme)
CUSTOM_CSS = """
<style>
    /* Base theme adjustments */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header card */
    .yh-header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .yh-header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .yh-header-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.25rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .text-emerald { color: #10b981; }
    .text-rose { color: #f43f5e; }
    .text-amber { color: #f59e0b; }
    .text-blue { color: #38bdf8; }

    /* Surface Card container */
    .surface-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
    }
    .surface-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.8rem;
    }

    /* Status badge pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-waiting {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .pill-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "raw_df": None,
        "dataset_name": None,
        "is_demo": False,
        "column_mapping": {},
        "cleaned_df": None,
        "summary": None,
        "trained": None,
        "opps_df": None,
        "analysis_warnings": [],
        "analysis_run": False,
        "active_section": "Dashboard",
        "uploader_key_counter": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# -----------------------------------------------------------------------------
# Chart Styling Helper
# -----------------------------------------------------------------------------
def style_plot(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=height,
        font=dict(family="Inter, -apple-system, sans-serif", color="#cbd5e1"),
        title=dict(
            text=title or "",
            font=dict(size=15, color="#f8fafc", family="Inter, -apple-system, sans-serif"),
            x=0.01,
            y=0.96,
        ) if title else None,
        margin=dict(l=45, r=30, t=50 if title else 25, b=45),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    return fig


# -----------------------------------------------------------------------------
# Column Auto-Detection Heuristics
# -----------------------------------------------------------------------------
COLUMN_SYNONYMS = {
    "Yield": ["yield", "yield%", "yield_pct", "yield_percentage", "lot_yield", "wafer_yield", "output_yield", "recovery"],
    "Temperature": ["temperature", "temp", "temp_c", "temperature_c", "temp_deg", "furnace_temp", "chamber_temp"],
    "Pressure": ["pressure", "press", "chamber_pressure", "pressure_torr", "press_bar", "vacuum_pressure"],
    "Power": ["power", "rf_power", "heater_power", "watts", "rf_watts", "generator_power"],
    "Defects": ["defects", "defect", "defect_count", "particles", "particle_count", "errors", "defect_density"],
    "GasFlow": ["gasflow", "gas_flow", "flow", "gas_sccm", "sccm", "flow_rate"],
    "Lot": ["lot", "lot_id", "wafer_lot", "batch", "batch_id", "id", "run_id"],
}


def guess_column_match(target_field: str, available_cols: List[str]) -> str:
    """Best effort column mapping guess based on normalized name matching."""
    norm_available = {c.lower().replace("_", "").replace(" ", "").replace("-", ""): c for c in available_cols}

    synonyms = COLUMN_SYNONYMS.get(target_field, [target_field.lower()])

    # 1. Exact match among synonyms
    for syn in synonyms:
        norm_syn = syn.lower().replace("_", "").replace(" ", "").replace("-", "")
        if norm_syn in norm_available:
            return norm_available[norm_syn]

    # 2. Substring match on normalized strings
    for norm_c, orig_c in norm_available.items():
        for syn in synonyms:
            norm_syn = syn.lower().replace("_", "").replace(" ", "").replace("-", "")
            if len(norm_syn) >= 3 and (norm_syn in norm_c or norm_c in norm_syn):
                return orig_c

    return "[Select Column]"


# -----------------------------------------------------------------------------
# Data Quality Assessment
# -----------------------------------------------------------------------------
def run_data_quality_audit(raw_df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
    total_rows = len(raw_df)
    total_cols = len(raw_df.columns)
    dup_rows = int(raw_df.duplicated().sum())

    missing_counts: Dict[str, int] = {}
    invalid_num_counts: Dict[str, int] = {}
    unmapped_required = []

    for req in REQUIRED_COLUMNS:
        col = mapping.get(req)
        if not col or col == "[Select Column]" or col not in raw_df.columns:
            unmapped_required.append(req)
        else:
            nans = int(raw_df[col].isna().sum())
            missing_counts[req] = nans
            coerced = pd.to_numeric(raw_df[col], errors="coerce")
            invalid_num = int(coerced.isna().sum() - nans)
            invalid_num_counts[req] = invalid_num

    # Calculate valid rows count if required columns mapped
    valid_rows = 0
    if not unmapped_required:
        subset_cols = [mapping[r] for r in REQUIRED_COLUMNS]
        temp_df = raw_df[subset_cols].copy()
        for c in temp_df.columns:
            temp_df[c] = pd.to_numeric(temp_df[c], errors="coerce")
        valid_rows = len(temp_df.dropna())

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "valid_rows": valid_rows,
        "dup_rows": dup_rows,
        "missing_counts": missing_counts,
        "invalid_num_counts": invalid_num_counts,
        "unmapped_required": unmapped_required,
        "is_schema_ready": len(unmapped_required) == 0,
    }


# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <div class="yh-header-card">
            <div class="yh-header-title">
                <span>⚡ YIELD HUNTER</span>
            </div>
            <div class="yh-header-subtitle">
                Intelligent Yield & Opportunity Analysis Platform — Precision Root Cause Diagnostics & Excursion Risk Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Sidebar Component
# -----------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ⚡ **YIELD HUNTER**")
        st.caption("Semiconductor Process Analytics & Yield Optimization")
        st.markdown("---")

        # Dataset Status Widget
        st.markdown("**Dataset Status**")
        if st.session_state["raw_df"] is not None:
            dataset_name = st.session_state["dataset_name"]
            row_count = len(st.session_state["raw_df"])
            status_text = "Analysis Ready" if st.session_state["analysis_run"] else "Dataset Loaded"
            pill_class = "pill-active" if st.session_state["analysis_run"] else "pill-info"

            st.markdown(
                f"""
                <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:0.8rem; margin-bottom:1rem;">
                    <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Active Dataset</div>
                    <div style="font-size:0.95rem; font-weight:700; color:#f8fafc; margin-top:0.2rem; word-break:break-all;">{dataset_name}</div>
                    <div style="margin-top:0.5rem; display:flex; justify-content:space-between; align-items:center;">
                        <span class="status-pill {pill_class}">● {status_text}</span>
                        <span style="font-size:0.8rem; color:#94a3b8;">{row_count:,} rows</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:0.8rem; margin-bottom:1rem;">
                    <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Active Dataset</div>
                    <div style="font-size:0.95rem; font-weight:600; color:#94a3b8; margin-top:0.2rem;">No data loaded</div>
                    <div style="margin-top:0.5rem;">
                        <span class="status-pill pill-waiting">Waiting for CSV</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Navigation
        st.markdown("**Navigation**")
        nav_options = [
            "Dashboard",
            "Data Input",
            "Analysis",
            "Results",
            "Future Lot Prediction",
            "About",
        ]
        icons = {
            "Dashboard": "🏠",
            "Data Input": "📥",
            "Analysis": "📊",
            "Results": "🎯",
            "Future Lot Prediction": "⚠️",
            "About": "ℹ️",
        }
        selected = st.radio(
            "Select View",
            nav_options,
            format_func=lambda x: f"{icons[x]} {x}",
            index=nav_options.index(st.session_state["active_section"]),
            label_visibility="collapsed",
        )
        st.session_state["active_section"] = selected

        st.markdown("---")

        # Quick Actions in Sidebar
        if st.session_state["raw_df"] is not None:
            if st.button("🔄 Reset / Clear Dataset", use_container_width=True):
                st.session_state["raw_df"] = None
                st.session_state["dataset_name"] = None
                st.session_state["is_demo"] = False
                st.session_state["column_mapping"] = {}
                st.session_state["cleaned_df"] = None
                st.session_state["summary"] = None
                st.session_state["trained"] = None
                st.session_state["opps_df"] = None
                st.session_state["analysis_warnings"] = []
                st.session_state["analysis_run"] = False
                st.session_state["uploader_key_counter"] += 1
                st.rerun()

        st.caption("Yield Hunter v2.0 • Production Analytics Edition")
        st.caption("IBM Bob AI Innovation Hackathon")

    return selected


# -----------------------------------------------------------------------------
# Section: Data Input & Quality
# -----------------------------------------------------------------------------
def render_data_input() -> None:
    st.markdown("### 📥 Data Input & Schema Configuration")
    st.caption("Upload your semiconductor fabrication wafer-lot CSV dataset, or load the verified demo dataset.")

    # Top Action Buttons: Demo Data & Template Download
    col_a, col_b, col_c = st.columns([1.5, 1.5, 3])
    with col_a:
        if st.button("🔬 Try Demo Dataset", use_container_width=True, type="secondary"):
            try:
                demo_df = pd.read_csv("wafer_data.csv")
                st.session_state["raw_df"] = demo_df
                st.session_state["dataset_name"] = "wafer_data.csv (Demo Dataset)"
                st.session_state["is_demo"] = True
                # Reset analysis run so user executes with button
                st.session_state["analysis_run"] = False
                # Auto map standard columns
                mapping = {col: col for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if col in demo_df.columns}
                if "Lot" in demo_df.columns:
                    mapping["Lot"] = "Lot"
                st.session_state["column_mapping"] = mapping
                st.success("Loaded verified Demo Dataset (500 wafer lots). Ready for analysis!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading demo dataset: {str(e)}")

    with col_b:
        template_csv = generate_template_csv()
        st.download_button(
            label="📄 Download CSV Template",
            data=template_csv,
            file_name="yield_hunter_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a properly formatted wafer lot CSV template with sample data.",
        )

    # Real Functional CSV Uploader
    st.markdown("#### 📁 Upload Wafer Lot CSV")
    uploaded_file = st.file_uploader(
        "Select a .csv file from your computer",
        type=["csv"],
        key=f"csv_uploader_{st.session_state['uploader_key_counter']}",
        help="Upload CSV containing process parameters (Temperature, Pressure, Power, Defects, Yield)",
    )

    if uploaded_file is not None:
        try:
            # Read CSV with pandas safely
            raw_csv_df = pd.read_csv(uploaded_file)
            if raw_csv_df.empty:
                st.error("The uploaded CSV file is empty. Please upload a CSV containing data rows.")
            else:
                st.session_state["raw_df"] = raw_csv_df
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["is_demo"] = False
                st.session_state["analysis_run"] = False
                st.success(f"Your CSV was uploaded successfully: **{uploaded_file.name}** ({len(raw_csv_df):,} rows, {len(raw_csv_df.columns)} columns)")
        except Exception as e:
            st.error(f"Failed to parse CSV file: {str(e)}. Please check that the file is a valid, uncorrupted CSV.")

    raw_df = st.session_state.get("raw_df")

    if raw_df is None:
        st.info("👈 Please upload a CSV file above or click **'🔬 Try Demo Dataset'** to begin.")
        return

    # Dataset Metadata Banner
    st.markdown("---")
    badge_label = "Demo Dataset" if st.session_state["is_demo"] else "User Uploaded CSV"
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#1e293b; border:1px solid #334155; border-radius:8px; padding:0.75rem 1.2rem; margin-bottom:1rem;">
            <div>
                <span style="font-weight:700; color:#f8fafc; font-size:1.05rem;">{st.session_state['dataset_name']}</span>
                <span style="margin-left:0.6rem; font-size:0.75rem; background:#334155; color:#38bdf8; padding:0.2rem 0.6rem; border-radius:4px; font-weight:600;">{badge_label}</span>
            </div>
            <div style="font-size:0.85rem; color:#94a3b8;">
                <strong>{len(raw_df):,}</strong> Total Rows &nbsp;|&nbsp; <strong>{len(raw_df.columns)}</strong> Columns
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data Preview
    st.markdown("#### 👁️ Data Preview (First 10 Rows)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    # Column Mapping Section
    st.markdown("---")
    st.markdown("#### 🗺️ Column Mapping")
    st.caption("Map your dataset's columns to the standard fields required by the Yield Hunter analytics engine.")

    available_cols = ["[Select Column]"] + list(raw_df.columns)
    mapping = dict(st.session_state.get("column_mapping", {}))

    # Render mapping inputs in clean 3-column layout
    col1, col2, col3 = st.columns(3)
    target_fields = [
        ("Yield", "Target Yield (%)", True),
        ("Defects", "Defect / Particle Count", True),
        ("Temperature", "Process Temperature", True),
        ("Pressure", "Chamber Pressure", True),
        ("Power", "RF / Process Power", True),
        ("GasFlow", "Gas Flow Rate (sccm)", False),
        ("Lot", "Lot / Batch Identifier", False),
    ]

    for idx, (target, label_desc, is_required) in enumerate(target_fields):
        target_col = [col1, col2, col3][idx % 3]
        with target_col:
            # Determine initial selection
            current_choice = mapping.get(target)
            if not current_choice or current_choice not in available_cols:
                current_choice = guess_column_match(target, list(raw_df.columns))

            default_idx = available_cols.index(current_choice) if current_choice in available_cols else 0
            req_tag = "🔴 Required" if is_required else "⚪ Optional"

            selected_col = st.selectbox(
                f"**{target}** ({req_tag})",
                available_cols,
                index=default_idx,
                help=f"{label_desc}. Required by algorithm." if is_required else f"{label_desc}.",
                key=f"map_select_{target}",
            )
            mapping[target] = selected_col

    st.session_state["column_mapping"] = mapping

    # Check Required Mapping Status
    unmapped = [r for r in REQUIRED_COLUMNS if mapping.get(r) in (None, "[Select Column]")]
    if unmapped:
        st.warning(f"⚠️ Your CSV is missing required column mapping(s): **{', '.join(unmapped)}**. Please select the matching columns above.")
    else:
        st.success("✅ All required columns are mapped and ready for analysis.")

    # Data Quality Expandable Section (Requirement 10)
    audit = run_data_quality_audit(raw_df, mapping)
    with st.expander("🔍 Data Quality Audit & Schema Inspection", expanded=True):
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Total Records", f"{audit['total_rows']:,}")
        q2.metric("Valid Clean Rows", f"{audit['valid_rows']:,}")
        q3.metric("Duplicate Rows", f"{audit['dup_rows']:,}")
        total_miss = sum(audit["missing_counts"].values())
        q4.metric("Missing Values", f"{total_miss:,}")
        total_inv = sum(audit["invalid_num_counts"].values())
        q5.metric("Invalid Numeric", f"{total_inv:,}")

        st.markdown("##### Required Columns Status")
        status_cols = st.columns(len(REQUIRED_COLUMNS))
        for i, req in enumerate(REQUIRED_COLUMNS):
            with status_cols[i]:
                src = mapping.get(req)
                if src and src != "[Select Column]":
                    miss = audit["missing_counts"].get(req, 0)
                    inv = audit["invalid_num_counts"].get(req, 0)
                    if miss == 0 and inv == 0:
                        st.markdown(f"**{req}**\n\n🟢 Valid (`{src}`)")
                    else:
                        st.markdown(f"**{req}**\n\n🟡 `{src}` ({miss} null, {inv} bad)")
                else:
                    st.markdown(f"**{req}**\n\n🔴 Not Mapped")

    # Analysis Trigger Button (Requirement 5)
    st.markdown("---")
    col_run, _ = st.columns([2, 3])
    with col_run:
        can_run = len(unmapped) == 0
        btn_label = "🚀 Run Yield Analysis" if can_run else "⚠️ Map Required Columns to Run"
        if st.button(btn_label, type="primary", use_container_width=True, disabled=not can_run):
            run_yield_analysis(raw_df, mapping)


# -----------------------------------------------------------------------------
# Analysis Execution Pipeline
# -----------------------------------------------------------------------------
def run_yield_analysis(raw_df: pd.DataFrame, mapping: Dict[str, str]) -> None:
    try:
        with st.spinner("Analyzing your dataset... Fitting ensemble models and diagnosing excursions..."):
            # Build transformed dataframe matching standard names
            mapped_data = {}
            for target in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
                src_col = mapping.get(target)
                if src_col and src_col != "[Select Column]" and src_col in raw_df.columns:
                    mapped_data[target] = raw_df[src_col]

            if mapping.get("Lot") and mapping.get("Lot") != "[Select Column]" and mapping.get("Lot") in raw_df.columns:
                mapped_data["Lot"] = raw_df[mapping["Lot"]]

            working_df = pd.DataFrame(mapped_data)

            # Clean and validate using existing engine logic
            cleaned_df, warnings = load_and_validate(working_df)

            if cleaned_df.empty:
                st.error("No valid numeric rows found after data cleaning. Please verify your data columns.")
                return

            # Train model using existing engine logic
            trained = train_risk_model(cleaned_df)

            # Compute summary KPIs using existing engine logic
            summary = compute_summary(cleaned_df)

            # Compute lot opportunities & excursions
            opps_df = compute_lot_opportunities(cleaned_df, trained)

            # Store in session state
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["trained"] = trained
            st.session_state["summary"] = summary
            st.session_state["opps_df"] = opps_df
            st.session_state["analysis_warnings"] = warnings
            st.session_state["analysis_run"] = True

            st.success("Analysis completed successfully! Proceeding to Dashboard...")
            time.sleep(0.5)
            st.session_state["active_section"] = "Dashboard"
            st.rerun()

    except Exception as e:
        st.error(f"Analysis encountered an issue: {str(e)}. Please check your dataset values and column mappings.")


# -----------------------------------------------------------------------------
# Section: Dashboard (Executive Overview)
# -----------------------------------------------------------------------------
def render_dashboard() -> None:
    if not st.session_state.get("analysis_run") or st.session_state.get("summary") is None:
        st.info("👈 No analysis results available yet. Navigate to **'Data Input'** and click **'🚀 Run Yield Analysis'** to generate your dashboard.")
        col1, _ = st.columns([2, 4])
        with col1:
            if st.button("Go to Data Input", type="primary"):
                st.session_state["active_section"] = "Data Input"
                st.rerun()
        return

    summary = st.session_state["summary"]
    df = st.session_state["cleaned_df"]
    trained = st.session_state["trained"]
    opps_df = st.session_state["opps_df"]

    st.markdown("### 🏠 Executive Dashboard")
    st.caption(f"Comprehensive yield performance summary for dataset **{st.session_state['dataset_name']}**")

    # Show data cleaning warnings if any
    for w in st.session_state.get("analysis_warnings", []):
        st.warning(f"ℹ️ {w}")

    # Executive KPI Metric Cards (Requirement 7)
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Records</div>
                <div class="kpi-value text-blue">{summary['total_lots']:,}</div>
                <div class="kpi-sub" style="color:#94a3b8;">Wafer lots audited</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        avg_color = "text-emerald" if summary['average_yield'] >= 90 else ("text-amber" if summary['average_yield'] >= 80 else "text-rose")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Yield</div>
                <div class="kpi-value {avg_color}">{summary['average_yield']:.2f}%</div>
                <div class="kpi-sub" style="color:#94a3b8;">Median: {summary['median_yield']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Best Yield</div>
                <div class="kpi-value text-emerald">{summary['best_yield']:.2f}%</div>
                <div class="kpi-sub text-emerald">Lot: {summary['best_lot']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Opportunity Gap</div>
                <div class="kpi-value text-amber">+{summary['potential_opportunity']:.2f}%</div>
                <div class="kpi-sub text-amber">Potential: +{summary['improvement_pct']:.1f}% gain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Low-Yield Excursions</div>
                <div class="kpi-value text-rose">{summary['low_yield_lots']}</div>
                <div class="kpi-sub text-rose">&le; {summary['low_yield_threshold']:.2f}% (P25)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg Defects</div>
                <div class="kpi-value" style="color:#e2e8f0;">{summary['avg_defects']:.1f}</div>
                <div class="kpi-sub" style="color:#94a3b8;">Worst Lot: {summary['worst_lot']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Visualizations Row: Distribution + Root Cause preview
    c_left, c_right = st.columns([3, 2])

    with c_left:
        fig_dist = px.histogram(
            df,
            x="Yield",
            nbins=35,
            marginal="box",
            title="Yield Distribution Across Wafer Lots",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_dist.add_vline(
            x=summary["low_yield_threshold"],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Excursion Threshold ({summary['low_yield_threshold']}%)",
            annotation_position="top left",
        )
        fig_dist.add_vline(
            x=summary["average_yield"],
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Mean ({summary['average_yield']}%)",
            annotation_position="top right",
        )
        style_plot(fig_dist, "Yield Distribution & Excursion Boundary", height=380)
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_right:
        imp_df = trained["importance_df"]
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Factor",
            orientation="h",
            text=imp_df["Importance"].round(1).astype(str) + "%",
            title="Top Factors Influencing Yield Variation",
            color="Importance",
            color_continuous_scale="Tealgrn",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        fig_imp.update_traces(textposition="outside")
        style_plot(fig_imp, "Root Cause Attribution (Feature Importance)", height=380)
        st.plotly_chart(fig_imp, use_container_width=True)

    # Top Opportunities Preview Table
    st.markdown("---")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("#### 🎯 Priority Yield Recovery Opportunities (Top 5 Lots)")
    with col_t2:
        if st.button("View Full Opportunities Table ➔", use_container_width=True):
            st.session_state["active_section"] = "Results"
            st.rerun()

    preview_cols = ["Lot", "Yield (%)", "Opportunity Gap (%)", "Risk Tier", "Primary Root Cause", "Recommended Action"]
    st.dataframe(opps_df[preview_cols].head(5), use_container_width=True)


# -----------------------------------------------------------------------------
# Section: Analysis (Charts & Deep Exploration)
# -----------------------------------------------------------------------------
def render_analysis() -> None:
    if not st.session_state.get("analysis_run") or st.session_state.get("cleaned_df") is None:
        st.info("👈 Please execute yield analysis first via **'Data Input'** to view deep parameter analytics.")
        return

    df = st.session_state["cleaned_df"]
    summary = st.session_state["summary"]
    features = get_feature_columns(df)

    st.markdown("### 📊 Process Parameter & Yield Analytics")
    st.caption("Investigate relationships, correlation structures, and parameter excursions against yield.")

    # Chronological / Lot Trendline
    if "Lot" in df.columns:
        st.markdown("#### 📈 Lot Yield Trendline")
        fig_time = px.line(
            df,
            x="Lot",
            y="Yield",
            title="Wafer Lot Yield Trendline Over Run Sequence",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_time.add_hline(
            y=summary["low_yield_threshold"],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Low Yield Cutoff ({summary['low_yield_threshold']}%)",
        )
        fig_time.add_hline(
            y=summary["average_yield"],
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Average Yield ({summary['average_yield']}%)",
        )
        style_plot(fig_time, "Yield Performance Across Manufacturing Sequence", height=350)
        st.plotly_chart(fig_time, use_container_width=True)

    # Interactive Parameter Explorer
    st.markdown("---")
    st.markdown("#### 🔬 Parameter vs. Yield Scatter Analysis")
    selected_param = st.selectbox("Select Process Parameter to Analyze:", features, index=0)

    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        # Scatter plot with trendline
        try:
            fig_scatter = px.scatter(
                df,
                x=selected_param,
                y="Yield",
                color="Defects",
                trendline="ols",
                title=f"{selected_param} vs. Yield (Color: Defects)",
                color_continuous_scale="Plasma",
                hover_data=["Lot"] if "Lot" in df.columns else None,
            )
        except Exception:
            fig_scatter = px.scatter(
                df,
                x=selected_param,
                y="Yield",
                color="Defects",
                title=f"{selected_param} vs. Yield (Color: Defects)",
                color_continuous_scale="Plasma",
                hover_data=["Lot"] if "Lot" in df.columns else None,
            )
        style_plot(fig_scatter, f"{selected_param} Impact on Lot Yield", height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_s2:
        # Distribution of the selected parameter
        fig_param_dist = px.histogram(
            df,
            x=selected_param,
            nbins=30,
            title=f"Distribution of {selected_param}",
            color_discrete_sequence=["#a855f7"],
        )
        style_plot(fig_param_dist, f"{selected_param} Operating Range", height=400)
        st.plotly_chart(fig_param_dist, use_container_width=True)

    # Correlation Matrix
    st.markdown("---")
    st.markdown("#### 🧬 Parameter Correlation Matrix")
    corr_cols = features + ["Yield"]
    corr_matrix = df[corr_cols].corr().round(2)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Pearson Correlation Heatmap (Process Factors & Yield)",
    )
    style_plot(fig_corr, "Process Factors vs Yield Correlation Matrix", height=420)
    st.plotly_chart(fig_corr, use_container_width=True)


# -----------------------------------------------------------------------------
# Section: Results & Opportunities
# -----------------------------------------------------------------------------
def render_results() -> None:
    if not st.session_state.get("analysis_run") or st.session_state.get("opps_df") is None:
        st.info("👈 Please execute yield analysis first via **'Data Input'** to view actionable results.")
        return

    opps_df = st.session_state["opps_df"]
    summary = st.session_state["summary"]

    st.markdown("### 🎯 Priority Yield Recovery Opportunities")
    st.caption("Ranked lots with actionable root-cause diagnostics and parameter excursion recovery paths.")

    # Filter by Risk Tier
    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 2, 2])
    with filter_col1:
        risk_options = ["All Lots", "High Risk Only", "Medium & High Risk", "Optimal Only"]
        selected_filter = st.selectbox("Filter by Risk Tier:", risk_options, index=0)

    if selected_filter == "High Risk Only":
        filtered_df = opps_df[opps_df["Risk Tier"] == "High Risk"]
    elif selected_filter == "Medium & High Risk":
        filtered_df = opps_df[opps_df["Risk Tier"].isin(["High Risk", "Medium Risk"])]
    elif selected_filter == "Optimal Only":
        filtered_df = opps_df[opps_df["Risk Tier"] == "Optimal"]
    else:
        filtered_df = opps_df

    with filter_col3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        # Download Results CSV (Requirement 9)
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name=f"yield_hunter_opportunities_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the complete analyzed results including diagnosed excursion causes and corrective actions.",
        )

    # Opportunity Summary Metrics
    st.markdown(
        f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:1rem; display:flex; gap:2rem; align-items:center;">
            <div><span style="color:#94a3b8; font-size:0.85rem;">Showing Lots:</span> <strong>{len(filtered_df):,}</strong></div>
            <div><span style="color:#94a3b8; font-size:0.85rem;">High Risk Excursions:</span> <strong style="color:#f43f5e;">{(opps_df['Risk Tier'] == 'High Risk').sum()}</strong></div>
            <div><span style="color:#94a3b8; font-size:0.85rem;">Mean Opportunity Gap:</span> <strong style="color:#fbbf24;">+{filtered_df['Opportunity Gap (%)'].mean():.2f}%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Results Table
    st.dataframe(filtered_df, use_container_width=True)

    # Risk Tier Breakdown Visualization
    st.markdown("---")
    st.markdown("#### 📊 Risk Tier Breakdown")
    tier_counts = opps_df["Risk Tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Lots"]

    color_map = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Optimal": "#10b981"}
    fig_pie = px.pie(
        tier_counts,
        names="Risk Tier",
        values="Lots",
        color="Risk Tier",
        color_discrete_map=color_map,
        hole=0.45,
        title="Lot Population by Risk Category",
    )
    style_plot(fig_pie, "Manufacturing Population Risk Breakdown", height=350)
    st.plotly_chart(fig_pie, use_container_width=True)


# -----------------------------------------------------------------------------
# Section: Future Lot Prediction (Requirement & Core Logic Preservation)
# -----------------------------------------------------------------------------
def render_future_lot_prediction() -> None:
    if not st.session_state.get("analysis_run") or st.session_state.get("trained") is None:
        st.info("👈 Please execute yield analysis first via **'Data Input'** to train the prediction model.")
        return

    df = st.session_state["cleaned_df"]
    trained = st.session_state["trained"]
    features = trained["features"]

    st.markdown("### ⚠️ Future Lot Risk & Yield Prediction")
    st.caption("Simulate an upcoming manufacturing lot before dispatch to detect potential excursions and prevent yield losses.")

    st.markdown("#### Input Planned Process Parameters")
    defaults = {f: float(df[f].median()) for f in features}
    input_values = {}

    cols = st.columns(len(features))
    for i, feat in enumerate(features):
        with cols[i]:
            input_values[feat] = st.number_input(
                f"{feat}",
                value=round(defaults[feat], 2),
                format="%.2f",
                key=f"predict_input_{feat}",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Lot Risk & Performance", type="primary"):
        try:
            result = predict_new_lot(trained, input_values)
            predicted_yield = result["predicted_yield"]
            risk = result["risk"]
            findings = result["findings"]

            risk_cfg = {
                "LOW": ("🟢", "#10b981", "Optimal Operating Regime"),
                "MEDIUM": ("🟡", "#f59e0b", "Moderate Excursion Risk"),
                "HIGH": ("🔴", "#ef4444", "Critical Excursion Risk"),
            }
            icon, color, label = risk_cfg.get(risk, ("⚪", "#94a3b8", "Unknown"))

            st.markdown(
                f"""
                <div style="background:#1e293b; border:2px solid {color}; border-radius:12px; padding:1.5rem; margin-top:1rem;">
                    <div style="font-size:0.9rem; font-weight:600; text-transform:uppercase; color:#94a3b8;">Prediction Result</div>
                    <div style="font-size:2rem; font-weight:800; color:{color}; margin-top:0.3rem;">
                        {icon} {risk} RISK &nbsp;—&nbsp; <span style="font-size:1.4rem; color:#f8fafc;">Predicted Yield: <strong>{predicted_yield:.2f}%</strong></span>
                    </div>
                    <div style="color:#94a3b8; font-size:0.9rem; margin-top:0.3rem;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if findings:
                st.markdown("#### ⚠️ Abnormal Parameter Diagnostics (Z-Score Deviation)")
                for f in findings:
                    st.warning(f"**{f['factor']}** is abnormally **{f['direction'].upper()}** (Z-Score: `{f['z_score']:+.2f}`). {f['message']}")
                    with st.expander(f"🛠️ Recommended Corrective Action for {f['factor']}"):
                        for action in f["actions"]:
                            st.write(f"→ **{action}**")
            else:
                st.success("✅ All planned parameters are within the healthy operating baseline. No excursion red flags detected.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# -----------------------------------------------------------------------------
# Section: About Platform
# -----------------------------------------------------------------------------
def render_about() -> None:
    st.markdown("### ℹ️ About Yield Hunter Platform")
    st.markdown(
        """
        **Yield Hunter** is an enterprise-grade semiconductor decision-support platform designed to automate
        wafer lot yield root-cause diagnostics, identify yield recovery opportunities, and predict risk for future runs.

        ---

        #### 🏗️ Architecture & Scientific Methodology
        1. **Association Ranking vs. Causality**:
           Yield Hunter trains a Random Forest ensemble to identify factors with the highest statistical association to yield variability.
           This ranking highlights parameters most strongly correlated with yield loss without making unverified causal assumptions.
        2. **Transparent Rule-Based Diagnostics**:
           Unlike opaque black-box AI tools, Yield Hunter evaluates deviations against a historical baseline derived from the fab's top-performing lots (75th percentile).
           Parameters exceeding 1.5 standard deviations trigger actionable, fab-validated recommendations.
        3. **Opportunity Sizing**:
           Every low-yield lot is benchmarked against optimal lot performance, calculating exact recovery potential and identifying the primary excursion driver.

        ---

        #### 📦 Tech Stack
        - **Core Engine**: Python 3.10+, Pandas, NumPy, Scikit-Learn (`RandomForestRegressor`), Statsmodels
        - **Visual Analytics**: Plotly Express & Plotly Graph Objects
        - **Deployment**: Streamlit Cloud Native
        """
    )


# -----------------------------------------------------------------------------
# Main Router
# -----------------------------------------------------------------------------
def main() -> None:
    render_header()
    active_view = render_sidebar()

    if active_view == "Dashboard":
        render_dashboard()
    elif active_view == "Data Input":
        render_data_input()
    elif active_view == "Analysis":
        render_analysis()
    elif active_view == "Results":
        render_results()
    elif active_view == "Future Lot Prediction":
        render_future_lot_prediction()
    elif active_view == "About":
        render_about()


if __name__ == "__main__":
    main()
