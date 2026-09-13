"""
app.py
Wafer Yield Root-Cause & Risk Prediction Dashboard
IBM Bob AI Hackathon prototype

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    load_and_validate,
    compute_summary,
    train_risk_model,
    predict_new_lot,
    get_feature_columns,
)

st.set_page_config(page_title="Wafer Yield Analyzer", layout="wide", page_icon="🏭")

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.title("🏭 Wafer Yield Analyzer")
st.sidebar.caption("Root-cause analysis & risk prediction for wafer lots")

uploaded_file = st.sidebar.file_uploader("Upload wafer lot CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo dataset instead", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Expected columns:**\n"
    "`Lot` (optional), `Temperature`, `Pressure`, `Power`, `Defects`, `Yield`, "
    "`GasFlow` (optional)"
)

if uploaded_file is not None and not use_demo:
    raw_df = pd.read_csv(uploaded_file)
elif use_demo:
    raw_df = pd.read_csv("wafer_data.csv")
else:
    st.info("Upload a CSV or check 'Use demo dataset' in the sidebar to begin.")
    st.stop()

df, warnings = load_and_validate(raw_df)
for w in warnings:
    st.sidebar.warning(w)

if df.empty:
    st.error("No valid rows found after cleaning. Please check your CSV.")
    st.stop()

# Train model once per dataset (cached on the dataframe's content)
@st.cache_resource(show_spinner="Training model on historical data...")
def get_trained_model(data: pd.DataFrame):
    return train_risk_model(data)

trained = get_trained_model(df)
summary = compute_summary(df)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📊 Yield & Defect Analysis", "🎯 Root Cause", "⚠️ Future Lot Prediction"],
)

# ---------------------------------------------------------------------------
# PAGE 1: Dashboard
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.caption("High-level overview of wafer lot yield performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Lots", summary["total_lots"])
    c2.metric("Average Yield", f"{summary['average_yield']}%")
    c3.metric("Low-Yield Lots", summary["low_yield_lots"],
              help=f"Lots at or below the 25th percentile ({summary['low_yield_threshold']}%)")
    c4.metric("Worst Lot", f"{summary['worst_lot']} ({summary['worst_yield']}%)")

    st.markdown("---")
    st.subheader("Yield distribution")
    fig = px.histogram(df, x="Yield", nbins=30, title="Distribution of Lot Yield")
    fig.add_vline(x=summary["low_yield_threshold"], line_dash="dash", line_color="red",
                  annotation_text="Low-yield threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent lots")
    st.dataframe(df.tail(15), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2: Yield & Defect Analysis
# ---------------------------------------------------------------------------
elif page == "📊 Yield & Defect Analysis":
    st.title("📊 Yield & Defect Analysis")
    st.caption("Explore how process parameters relate to yield")

    if "Lot" in df.columns:
        st.subheader("Yield over lots (proxy for time)")
        fig_time = px.line(df, x="Lot", y="Yield", markers=False, title="Yield across lots")
        fig_time.add_hline(y=summary["low_yield_threshold"], line_dash="dash", line_color="red")
        st.plotly_chart(fig_time, use_container_width=True)

    features = get_feature_columns(df)
    st.subheader("Parameter vs Yield")
    cols = st.columns(2)
    for i, feat in enumerate(features):
        with cols[i % 2]:
            fig = px.scatter(df, x=feat, y="Yield", color="Defects",
                              trendline="ols", title=f"{feat} vs Yield")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation matrix")
    corr_cols = features + ["Yield"]
    corr = df[corr_cols].corr().round(2)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                          title="Correlation between parameters and yield")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3: Root Cause
# ---------------------------------------------------------------------------
elif page == "🎯 Root Cause":
    st.title("🎯 Root Cause — Ranked Contributing Factors")
    st.caption(
        "These are parameters most **statistically associated** with yield variation "
        "(from feature importance of a trained model) — not a proven causal analysis."
    )

    imp_df = trained["importance_df"]
    fig = px.bar(imp_df, x="Importance", y="Factor", orientation="h",
                 text=imp_df["Importance"].round(1).astype(str) + "%",
                 title="Top contributing factors to yield variation")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What this means")
    for _, row in imp_df.iterrows():
        st.write(f"**{row['Factor']}** — {row['Importance']:.1f}% relative importance")

    st.markdown("---")
    st.subheader("Low-yield lots for investigation")
    low_lots = df[df["Yield"] <= summary["low_yield_threshold"]].sort_values("Yield")
    st.dataframe(low_lots, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 4: Future Lot Prediction
# ---------------------------------------------------------------------------
elif page == "⚠️ Future Lot Prediction":
    st.title("⚠️ Future Lot Risk Prediction")
    st.caption("Enter the planned process parameters for an upcoming lot to assess risk before running it.")

    features = trained["features"]
    input_values = {}

    cols = st.columns(len(features))
    defaults = {f: float(df[f].median()) for f in features}

    for i, feat in enumerate(features):
        with cols[i]:
            input_values[feat] = st.number_input(
                feat, value=round(defaults[feat], 2), format="%.2f"
            )

    if st.button("🔍 PREDICT", type="primary"):
        result = predict_new_lot(trained, input_values)

        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[result["risk"]]
        st.markdown(f"## {risk_color} {result['risk']} RISK")
        st.markdown(f"### Predicted Yield: **{result['predicted_yield']}%**")

        if result["findings"]:
            st.subheader("Why?")
            for f in result["findings"]:
                st.warning(f"**{f['factor']}** is abnormally **{f['direction']}** "
                           f"(z-score: {f['z_score']})")

            st.subheader("Recommended actions")
            for f in result["findings"]:
                with st.expander(f"⚠️ {f['factor']} — investigate"):
                    for action in f["actions"]:
                        st.write(f"→ {action}")
        else:
            st.success("All parameters are within normal historical range. No red flags found.")

st.sidebar.markdown("---")
st.sidebar.caption("IBM Bob AI Hackathon — Wafer Yield Prototype")
