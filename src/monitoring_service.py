# ============================================================
# MONITORING SERVICE
# ============================================================

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MONITORING_DIR = BASE_DIR / "monitoring_data"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    MONITORING_DIR / "predictions_log.csv"
)


# ============================================================
# 2. LOG DE PREVISÃO
# ============================================================

def log_prediction(
    client_features,
    prediction_result
):

    """
    Save one prediction event for future monitoring.
    """

    record = {}

    # Timestamp UTC
    record["timestamp"] = (
        datetime.now(timezone.utc).isoformat()
    )

    # Features usadas na requisição
    for key, value in client_features.items():
        record[f"feature__{key}"] = value

    # Saídas do modelo
    record["pd"] = prediction_result[
        "probability_of_default"
    ]

    record["risk_level"] = prediction_result[
        "risk_level"
    ]

    record["decision"] = prediction_result[
        "decision"
    ]

    record["expected_loss"] = prediction_result[
        "expected_loss"
    ]

    record["expected_profit"] = prediction_result[
        "expected_profit"
    ]

    new_row = pd.DataFrame([record])

    # Se o arquivo ainda não existe
    if not PREDICTIONS_FILE.exists():

        new_row.to_csv(
            PREDICTIONS_FILE,
            index=False
        )

    else:

        new_row.to_csv(
            PREDICTIONS_FILE,
            mode="a",
            header=False,
            index=False
        )


# ============================================================
# 3. CARREGAR LOG
# ============================================================

def load_prediction_log():

    if not PREDICTIONS_FILE.exists():

        return pd.DataFrame()

    return pd.read_csv(
        PREDICTIONS_FILE
    )


# ============================================================
# 4. RESUMO DE PRODUÇÃO
# ============================================================

def production_summary():

    df = load_prediction_log()

    if df.empty:

        return {
            "status": "NO_DATA",
            "predictions": 0
        }

    summary = {

        "status":
            "OK",

        "predictions":
            len(df),

        "mean_pd":
            float(df["pd"].mean()),

        "min_pd":
            float(df["pd"].min()),

        "max_pd":
            float(df["pd"].max()),

        "approve_rate":
            float(
                (df["decision"] == "APPROVE")
                .mean()
            ),

        "review_rate":
            float(
                (df["decision"] == "REVIEW")
                .mean()
            ),

        "reject_rate":
            float(
                (df["decision"] == "REJECT")
                .mean()
            ),

        "mean_expected_loss":
            float(
                df["expected_loss"].mean()
            ),

        "mean_expected_profit":
            float(
                df["expected_profit"].mean()
            )
    }

    return summary