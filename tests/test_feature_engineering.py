import pandas as pd
import numpy as np

from src.feature_engineering import create_features


def test_create_features():

    df = pd.DataFrame({
        "DAYS_BIRTH": [-3652.5],
        "DAYS_EMPLOYED": [-365.25],
        "AMT_CREDIT": [500000],
        "AMT_INCOME_TOTAL": [100000],
        "AMT_ANNUITY": [20000],
        "AMT_GOODS_PRICE": [450000]
    })

    result = create_features(df)

    assert np.isclose(result["AGE_YEARS"].iloc[0], 10)
    assert np.isclose(result["EMPLOYED_YEARS"].iloc[0], 1)

    assert np.isclose(
        result["CREDIT_INCOME_RATIO"].iloc[0],
        5
    )

    assert np.isclose(
        result["ANNUITY_INCOME_RATIO"].iloc[0],
        0.2
    )

    assert np.isclose(
        result["GOODS_CREDIT_RATIO"].iloc[0],
        0.9
    )
    
def test_days_employed_anomaly_becomes_nan():

    df = pd.DataFrame({
        "DAYS_BIRTH": [-15000],
        "DAYS_EMPLOYED": [365243],
        "AMT_CREDIT": [500000],
        "AMT_INCOME_TOTAL": [100000],
        "AMT_ANNUITY": [20000],
        "AMT_GOODS_PRICE": [450000]
    })

    result = create_features(df)

    assert pd.isna(result["EMPLOYED_YEARS"].iloc[0])