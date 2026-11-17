"""
tests/test_prediction.py
Unit tests for expense_prediction module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from modules.expense_prediction import (
    run_prediction, _build_monthly_expense_series, _build_features, FEATURE_COLS,
)


# ── Fixtures ──────────────────────────────────────────────────────

def _make_df(n_months: int, base_expense: float = 30000) -> pd.DataFrame:
    """Create a DataFrame with n_months of expense data."""
    rows = []
    for i in range(n_months):
        month = (i % 12) + 1
        year  = 2025 + (i // 12)
        rows.append({
            "date":        f"{year}-{month:02d}-15",
            "description": "Test expense",
            "amount":      base_expense + np.random.uniform(-3000, 3000),
            "type":        "expense",
            "category":    "Food & Dining",
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Monthly Aggregation ───────────────────────────────────────────

def test_build_monthly_expense_series():
    df = _make_df(6)
    monthly = _build_monthly_expense_series(df)
    assert len(monthly) == 6
    assert "expense" in monthly.columns
    assert "month" in monthly.columns


def test_build_monthly_expense_series_empty():
    df = pd.DataFrame(columns=["date","description","amount","type","category"])
    monthly = _build_monthly_expense_series(df)
    assert monthly.empty


# ── Feature Building ──────────────────────────────────────────────

def test_build_features_columns():
    df = _make_df(6)
    monthly = _build_monthly_expense_series(df)
    feat = _build_features(monthly)
    for col in FEATURE_COLS:
        assert col in feat.columns, f"Missing feature: {col}"


def test_build_features_sin_cos_range():
    df = _make_df(12)
    monthly = _build_monthly_expense_series(df)
    feat = _build_features(monthly)
    assert feat["sin_month"].between(-1, 1).all()
    assert feat["cos_month"].between(-1, 1).all()


def test_build_features_lag_shift():
    df = _make_df(6)
    monthly = _build_monthly_expense_series(df)
    feat = _build_features(monthly)
    # lag_1 of row 1 should equal expense of row 0
    assert feat.iloc[1]["lag_1"] == pytest.approx(monthly.iloc[0]["expense"], rel=1e-5)


# ── run_prediction ────────────────────────────────────────────────

def test_run_prediction_no_data():
    df = pd.DataFrame()
    result = run_prediction(df)
    assert result["status"] == "no_data"


def test_run_prediction_insufficient_data():
    df = _make_df(2)  # Only 2 months — below MIN_MONTHS=4
    result = run_prediction(df)
    assert result["status"] == "insufficient_data"


def test_run_prediction_success():
    df = _make_df(8)
    result = run_prediction(df)
    assert result["status"] == "success"
    assert result["next_month_prediction"] >= 0
    assert "metrics" in result
    assert "monthly_data" in result


def test_run_prediction_forecast_in_data():
    df = _make_df(8)
    result = run_prediction(df)
    monthly_data = result["monthly_data"]
    # Last row should be the forecast
    assert "forecast" in monthly_data["split_point"].values


def test_run_prediction_chronological_split():
    """Ensure test rows come AFTER train rows in the result data."""
    df = _make_df(10)
    result = run_prediction(df)
    monthly_data = result["monthly_data"]
    train_indices = monthly_data[monthly_data["split_point"] == "train"].index.tolist()
    test_indices  = monthly_data[monthly_data["split_point"] == "test"].index.tolist()
    if train_indices and test_indices:
        assert max(train_indices) < min(test_indices)


def test_run_prediction_no_negative_forecast():
    """Predicted amount should never be negative."""
    df = _make_df(6)
    result = run_prediction(df)
    if result["status"] == "success":
        assert result["next_month_prediction"] >= 0


def test_run_prediction_metrics_present():
    df = _make_df(8)
    result = run_prediction(df)
    if result["status"] == "success" and result["metrics"]:
        assert "mae" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert result["metrics"]["mae"] >= 0
        assert result["metrics"]["rmse"] >= 0
