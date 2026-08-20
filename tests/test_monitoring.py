import pandas as pd

from src.monitoring import (
    calculate_psi,
    classify_drift,
    feature_drift_report
)


def test_psi_same_distribution():

    reference = pd.Series(
        range(1, 101)
    )

    production = pd.Series(
        range(1, 101)
    )

    psi = calculate_psi(
        reference,
        production
    )

    assert psi < 0.10


def test_psi_different_distribution():

    reference = pd.Series(
        range(1, 101)
    )

    production = pd.Series(
        range(1000, 1100)
    )

    psi = calculate_psi(
        reference,
        production
    )

    assert psi >= 0.25


def test_classify_drift():

    assert classify_drift(0.05) == "STABLE"

    assert (
        classify_drift(0.15)
        == "MODERATE_DRIFT"
    )

    assert (
        classify_drift(0.30)
        == "SIGNIFICANT_DRIFT"
    )


def test_feature_drift_report():

    reference = pd.DataFrame({
        "feature_a": range(1, 101)
    })

    production = pd.DataFrame({
        "feature_a": range(1, 101)
    })

    report = feature_drift_report(
        reference,
        production
    )

    assert len(report) == 1

    assert (
        report.iloc[0]["status"]
        == "STABLE"
    )