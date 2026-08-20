# ============================================================
# MODEL MONITORING - DATA DRIFT + PREDICTION DRIFT
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 1. PSI
# Population Stability Index
# ============================================================

def calculate_psi(
    expected,
    actual,
    bins=10
):
    """
    Compare two numerical distributions.

    PSI interpretation:
        < 0.10   -> little/no drift
        0.10-0.25 -> moderate drift
        > 0.25   -> significant drift
    """

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Quantile bins based on reference data
    breakpoints = np.unique(
        np.quantile(
            expected,
            np.linspace(0, 1, bins + 1)
        )
    )
        # Permitir valores de produção fora
        # dos limites observados na referência
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # If there are too few unique values
    if len(breakpoints) < 3:
        return 0.0

    expected_bins = pd.cut(
        expected,
        bins=breakpoints,
        include_lowest=True
    )

    actual_bins = pd.cut(
        actual,
        bins=breakpoints,
        include_lowest=True
    )

    expected_pct = (
        expected_bins
        .value_counts(normalize=True, sort=False)
    )

    actual_pct = (
        actual_bins
        .value_counts(normalize=True, sort=False)
        .reindex(expected_pct.index, fill_value=0)
    )

    # Avoid division by zero / log(0)
    expected_pct = expected_pct.clip(lower=1e-6)
    actual_pct = actual_pct.clip(lower=1e-6)

    psi = (
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    ).sum()

    return float(psi)


# ============================================================
# 2. CLASSIFY DRIFT
# ============================================================

def classify_drift(psi_value):

    if pd.isna(psi_value):
        return "UNKNOWN"

    if psi_value < 0.10:
        return "STABLE"

    elif psi_value < 0.25:
        return "MODERATE_DRIFT"

    else:
        return "SIGNIFICANT_DRIFT"


# ============================================================
# 3. FEATURE DRIFT REPORT
# ============================================================

def feature_drift_report(
    reference_df,
    production_df,
    features=None
):

    if features is None:

        features = reference_df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    results = []

    for feature in features:

        if feature not in production_df.columns:
            continue

        psi = calculate_psi(
            reference_df[feature],
            production_df[feature]
        )

        results.append({
            "feature": feature,
            "psi": psi,
            "status": classify_drift(psi)
        })

    report = pd.DataFrame(results)

    if not report.empty:

        report = report.sort_values(
            "psi",
            ascending=False
        ).reset_index(drop=True)

    return report


# ============================================================
# 4. PREDICTION DRIFT
# ============================================================

def prediction_drift_report(
    reference_pd,
    production_pd
):

    psi = calculate_psi(
        reference_pd,
        production_pd
    )

    return {
        "reference_mean_pd":
            float(np.mean(reference_pd)),

        "production_mean_pd":
            float(np.mean(production_pd)),

        "psi":
            psi,

        "status":
            classify_drift(psi)
    }


# ============================================================
# 5. MONITORING SUMMARY
# ============================================================

def monitoring_summary(
    reference_df,
    production_df,
    reference_pd,
    production_pd,
    features=None
):

    feature_report = feature_drift_report(
        reference_df,
        production_df,
        features=features
    )

    prediction_report = prediction_drift_report(
        reference_pd,
        production_pd
    )

    significant_features = 0
    moderate_features = 0

    if not feature_report.empty:

        significant_features = (
            feature_report["status"]
            == "SIGNIFICANT_DRIFT"
        ).sum()

        moderate_features = (
            feature_report["status"]
            == "MODERATE_DRIFT"
        ).sum()

    return {

        "feature_drift": feature_report,

        "prediction_drift": prediction_report,

        "significant_feature_drift_count":
            int(significant_features),

        "moderate_feature_drift_count":
            int(moderate_features)
    }