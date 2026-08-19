import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # 1. Idade do cliente em anos
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    # 2. Indicador da anomalia em DAYS_EMPLOYED
    df["DAYS_EMPLOYED_ANOM"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype(int)

    # 3. Transformar o valor anômalo em NaN
    employed = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # 4. Tempo empregado em anos
    df["EMPLOYED_YEARS"] = -employed / 365.25

    # 5. Crédito em relação à renda
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    )

    # 6. Anuidade em relação à renda
    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    )

    # 7. Preço do bem em relação ao crédito
    df["GOODS_CREDIT_RATIO"] = (
        df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    )

    return df