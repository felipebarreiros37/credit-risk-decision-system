import pandas as pd


def validate_data(df: pd.DataFrame) -> None:

    required_columns = [
        "SK_ID_CURR",
        "TARGET",
        "DAYS_BIRTH",
    ]

    # 1. Colunas obrigatórias
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # 2. ID duplicado
    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("Duplicated SK_ID_CURR values found")

    # 3. TARGET válido
    valid_targets = {0, 1}

    if not set(df["TARGET"].dropna().unique()).issubset(valid_targets):
        raise ValueError("TARGET contains invalid values")

    # 4. SK_ID_CURR numérico
    if not pd.api.types.is_numeric_dtype(df["SK_ID_CURR"]):
        raise ValueError("SK_ID_CURR must be numeric")

    # 5. TARGET numérico
    if not pd.api.types.is_numeric_dtype(df["TARGET"]):
        raise ValueError("TARGET must be numeric")

    # 6. DAYS_BIRTH deve ser negativo
    if (df["DAYS_BIRTH"] >= 0).any():
        raise ValueError("DAYS_BIRTH contains invalid values")
    # 7. Colunas críticas não podem ter missing
    critical_columns = ["SK_ID_CURR", "TARGET"]

    for column in critical_columns:
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values")
    print("Data validation passed.")


if __name__ == "__main__":
    df = pd.read_csv("data/raw/application_train.csv")
    validate_data(df)
    # 8. Regras de domínio básicas

    if (df["AMT_CREDIT"] <= 0).any():
        raise ValueError("AMT_CREDIT must be positive")

    if (df["AMT_INCOME_TOTAL"] <= 0).any():
        raise ValueError("AMT_INCOME_TOTAL must be positive")

    if (df["DAYS_BIRTH"] > -3650).any():
        raise ValueError("DAYS_BIRTH contains implausible age values")