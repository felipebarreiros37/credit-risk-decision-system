# ============================================================
# CREDIT RISK - PRODUCTION PREDICTION ENGINE
# ============================================================

from pathlib import Path
import json
import joblib
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# ============================================================
# 2. LOAD ARTIFACTS
# ============================================================

model = joblib.load(
    ARTIFACTS_DIR / "xgboost_champion.joblib"
)

preprocessor = joblib.load(
    ARTIFACTS_DIR / "preprocessor.joblib"
)

calibrator = joblib.load(
    ARTIFACTS_DIR / "platt_calibrator.joblib"
)

top_features = joblib.load(
    ARTIFACTS_DIR / "top_75_features.joblib"
)


with open(
    ARTIFACTS_DIR / "model_config.json",
    "r"
) as f:

    config = json.load(f)


# ============================================================
# 3. ECONOMIC DECISION
# ============================================================

def economic_decision(pd_value, expected_profit):

    approve_threshold = config["approve_threshold"]

    if expected_profit <= 0:
        return "REJECT"

    elif pd_value < approve_threshold:
        return "APPROVE"

    else:
        return "REVIEW"


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_credit_risk(client_data):

    """
    Receives client data as dictionary or pandas DataFrame.

    Returns:
        Probability of Default
        Expected Loss
        Expected Revenue
        Expected Profit
        Expected Profit Margin
        Credit Decision
    """

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(client_data, dict):

        client_df = pd.DataFrame(
            [client_data]
        )

    elif isinstance(client_data, pd.DataFrame):

        client_df = client_data.copy()

    else:

        raise TypeError(
            "client_data must be a dictionary "
            "or pandas DataFrame."
        )


    # --------------------------------------------------------
    # Select model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in top_features
        if feature not in client_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )


    X_client = client_df[
        top_features
    ].copy()


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_processed = preprocessor.transform(
        X_client
    )


    # --------------------------------------------------------
    # Raw XGBoost probability
    # --------------------------------------------------------

    raw_pd = model.predict_proba(
        X_processed
    )[:, 1]


    # --------------------------------------------------------
    # Probability calibration
    # --------------------------------------------------------

    calibrated_pd = calibrator.predict_proba(
        raw_pd.reshape(-1, 1)
    )[:, 1]


    # --------------------------------------------------------
    # Economic calculations
    # --------------------------------------------------------

    results = []

    for i, pd_value in enumerate(calibrated_pd):

        # Exposure at Default
        ead = float(
            client_df.iloc[i]["AMT_CREDIT"]
        )

        lgd = config["lgd"]

        revenue_rate = config[
            "revenue_rate"
        ]

        funding_rate = config[
            "funding_cost_rate"
        ]

        operational_rate = config[
            "operational_cost_rate"
        ]


        # Expected Revenue
        expected_revenue = (
            ead
            * revenue_rate
            * (1 - pd_value)
        )


        # Expected Loss
        expected_loss = (
            pd_value
            * ead
            * lgd
        )


        # Funding Cost
        funding_cost = (
            ead
            * funding_rate
        )


        # Operational Cost
        operational_cost = (
            ead
            * operational_rate
        )


        # Expected Profit
        expected_profit = (
            expected_revenue
            - expected_loss
            - funding_cost
            - operational_cost
        )


        # Expected Profit Margin
        expected_profit_margin = (
            expected_profit / ead
            if ead > 0
            else 0
        )


        # Decision
        decision = economic_decision(
            pd_value,
            expected_profit
        )


        # Risk level
        if pd_value < 0.05:

            risk_level = "LOW"

        elif pd_value < 0.15:

            risk_level = "MEDIUM"

        else:

            risk_level = "HIGH"


        results.append({

            "probability_of_default":
                round(float(pd_value), 6),

            "risk_level":
                risk_level,

            "ead":
                round(ead, 2),

            "expected_revenue":
                round(expected_revenue, 2),

            "expected_loss":
                round(expected_loss, 2),

            "funding_cost":
                round(funding_cost, 2),

            "operational_cost":
                round(operational_cost, 2),

            "expected_profit":
                round(expected_profit, 2),

            "expected_profit_margin":
                round(
                    expected_profit_margin,
                    6
                ),

            "decision":
                decision
        })


    # --------------------------------------------------------
    # Single client -> dictionary
    # Multiple clients -> list
    # --------------------------------------------------------

    if len(results) == 1:

        return results[0]

    return results


# ============================================================
# 5. HEALTH CHECK
# ============================================================

def model_info():

    return {

        "model":
            "XGBoost",

        "calibration":
            config["calibration_method"],

        "test_roc_auc":
            config["model_test_roc_auc"],

        "validation_roc_auc":
            config["model_validation_roc_auc"],

        "number_of_features":
            len(top_features),

        "status":
            "ready"
    }