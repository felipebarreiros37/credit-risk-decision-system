import pandas as pd
import pytest

from src.data_validation import validate_data


def test_valid_data_passes():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "DAYS_BIRTH": [-10000, -15000]
    })

    validate_data(df)


def test_invalid_target_fails():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 99],
        "DAYS_BIRTH": [-10000, -15000]
    })

    with pytest.raises(ValueError, match="TARGET contains invalid values"):
        validate_data(df)


def test_duplicated_id_fails():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "TARGET": [0, 1],
        "DAYS_BIRTH": [-10000, -15000]
    })

    with pytest.raises(ValueError, match="Duplicated SK_ID_CURR values found"):
        validate_data(df)


def test_invalid_days_birth_fails():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "DAYS_BIRTH": [-10000, 15000]
    })

    with pytest.raises(ValueError, match="DAYS_BIRTH contains invalid values"):
        validate_data(df)

def test_missing_required_column_fails():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1]
    })

    with pytest.raises(ValueError, match="Missing required column: DAYS_BIRTH"):
        validate_data(df)

def test_missing_sk_id_fails():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, None],
        "TARGET": [0, 1],
        "DAYS_BIRTH": [-10000, -15000]
    })

    with pytest.raises(ValueError, match="SK_ID_CURR contains missing values"):
        validate_data(df)


def test_missing_target_fails():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, None],
        "DAYS_BIRTH": [-10000, -15000]
    })

    with pytest.raises(ValueError, match="TARGET contains missing values"):
        validate_data(df)