"""
engine.py
Core logic for the Wafer Yield Root-Cause & Prediction app.
Kept separate from the Streamlit UI (app.py) so it's easy to unit-test
and easy for judges to read.

Exposes:
    load_and_validate(df)              -> cleaned df, list of warnings
    compute_summary(df)                -> dict of dashboard KPIs
    compute_feature_importance(df)     -> DataFrame[factor, importance_pct]
    generate_recommendations(row, importance_df, baseline) -> list[str]
    train_risk_model(df)               -> (model, baseline_stats)
    predict_new_lot(model, baseline, input_dict) -> dict(risk, predicted_yield, reasons)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REQUIRED_COLUMNS = ["Temperature", "Pressure", "Power", "Defects", "Yield"]
OPTIONAL_COLUMNS = ["GasFlow"]

FEATURE_COLUMNS_DEFAULT = ["Temperature", "Pressure", "Power", "Defects", "GasFlow"]

LOW_YIELD_THRESHOLD_PERCENTILE = 0.25  # bottom 25% of yield = "low yield lot" for labeling


def load_and_validate(df: pd.DataFrame):
    """Check the uploaded CSV has the columns we need. Returns (df, warnings)."""
    warnings = []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        warnings.append(
            f"Missing required column(s): {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )
    if "Lot" not in df.columns:
        df.insert(0, "Lot", [f"L{i+1:03d}" for i in range(len(df))])

    # Coerce numeric columns
    for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[c for c in REQUIRED_COLUMNS if c in df.columns])
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"Dropped {dropped} row(s) with missing/invalid numeric values.")

    return df, warnings


def get_feature_columns(df: pd.DataFrame):
    return [c for c in FEATURE_COLUMNS_DEFAULT if c in df.columns]


def compute_summary(df: pd.DataFrame) -> dict:
    low_thresh = df["Yield"].quantile(LOW_YIELD_THRESHOLD_PERCENTILE)
    return {
        "total_lots": len(df),
        "average_yield": round(df["Yield"].mean(), 2),
        "low_yield_threshold": round(low_thresh, 2),
        "low_yield_lots": int((df["Yield"] <= low_thresh).sum()),
        "avg_defects": round(df["Defects"].mean(), 2),
        "worst_lot": df.loc[df["Yield"].idxmin(), "Lot"] if "Lot" in df.columns else None,
        "worst_yield": round(df["Yield"].min(), 2),
    }


def compute_feature_importance(df: pd.DataFrame):
    """
    Trains a RandomForest to predict Yield from process parameters, then
    reads off feature_importances_ as a proxy for 'which parameters are most
    strongly associated with low yield'. This is an ASSOCIATION ranking,
    not a proven causal analysis -- labeled accordingly in the UI.
    """
    features = get_feature_columns(df)
    X = df[features]
    y = df["Yield"]

    model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    model.fit(X, y)

    importances = model.feature_importances_
    pct = (importances / importances.sum()) * 100

    out = pd.DataFrame({"Factor": features, "Importance": pct})
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    return out, model


def _baseline_stats(df: pd.DataFrame) -> dict:
    """Median + std of each feature among the HIGH-yield (healthy) lots,
    used as the 'what does normal look like' reference for recommendations
    and for the future-lot predictor's explanation."""
    high_thresh = df["Yield"].quantile(0.75)
    healthy = df[df["Yield"] >= high_thresh]
    features = get_feature_columns(df)
    stats = {}
    for f in features:
        stats[f] = {
            "median": healthy[f].median(),
            "std": df[f].std(),  # use full-population std for a stable z-score
        }
    return stats


def generate_recommendations(input_values: dict, importance_df: pd.DataFrame,
                              baseline: dict, z_threshold: float = 1.5):
    """
    Simple, transparent RULE-BASED recommendations (no black-box AI needed here):
    for each top contributing factor, flag it if the new/lot's value is more
    than `z_threshold` standard deviations from the healthy-lot median.
    """
    RULES = {
        "Temperature": [
            "Check temperature sensor calibration",
            "Compare against furnace/chamber logs for drift",
            "Review recent PM (preventive maintenance) records",
        ],
        "Pressure": [
            "Check pressure-control system / regulator",
            "Inspect for chamber leaks",
            "Review process recipe pressure setpoints",
        ],
        "Power": [
            "Check RF/power supply calibration",
            "Compare against equipment matching network logs",
        ],
        "Defects": [
            "Run defect inspection / classification (SEM review)",
            "Compare defect map against known equipment signatures",
        ],
        "GasFlow": [
            "Check mass flow controller (MFC) calibration",
            "Inspect gas lines for blockage or leaks",
        ],
    }

    findings = []
    top_factors = importance_df["Factor"].tolist()

    for factor in top_factors:
        if factor not in input_values or factor not in baseline:
            continue
        median = baseline[factor]["median"]
        std = baseline[factor]["std"] or 1e-6
        z = (input_values[factor] - median) / std
        if abs(z) >= z_threshold:
            direction = "high" if z > 0 else "low"
            findings.append({
                "factor": factor,
                "direction": direction,
                "z_score": round(z, 2),
                "message": f"{factor} is abnormally {direction} "
                           f"(value={input_values[factor]}, healthy median={median:.1f})",
                "actions": RULES.get(factor, ["Investigate this parameter further"]),
            })

    return findings


def train_risk_model(df: pd.DataFrame):
    """Trains the yield-prediction model and returns it plus baseline stats
    and risk-band thresholds derived from the historical distribution."""
    importance_df, model = compute_feature_importance(df)
    baseline = _baseline_stats(df)

    yield_25 = df["Yield"].quantile(0.25)
    yield_50 = df["Yield"].quantile(0.50)

    thresholds = {
        "high_risk_below": round(yield_25, 2),   # predicted yield below this -> HIGH risk
        "medium_risk_below": round(yield_50, 2),  # below this (but above high) -> MEDIUM
    }

    return {
        "model": model,
        "importance_df": importance_df,
        "baseline": baseline,
        "thresholds": thresholds,
        "features": get_feature_columns(df),
    }


def predict_new_lot(trained, input_values: dict):
    """
    trained: dict returned by train_risk_model()
    input_values: dict like {"Temperature": 475, "Pressure": 2.9, "Power": 850, "Defects": 20, "GasFlow": 118}
    """
    model = trained["model"]
    features = trained["features"]
    thresholds = trained["thresholds"]

    X_new = pd.DataFrame([{f: input_values.get(f, np.nan) for f in features}])
    if X_new.isnull().any().any():
        missing = X_new.columns[X_new.isnull().any()].tolist()
        raise ValueError(f"Missing input value(s) for prediction: {missing}")

    predicted_yield = float(model.predict(X_new)[0])

    if predicted_yield <= thresholds["high_risk_below"]:
        risk = "HIGH"
    elif predicted_yield <= thresholds["medium_risk_below"]:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    findings = generate_recommendations(input_values, trained["importance_df"], trained["baseline"])

    return {
        "predicted_yield": round(predicted_yield, 2),
        "risk": risk,
        "findings": findings,
        "top_factors": trained["importance_df"]["Factor"].tolist()[:3],
    }
