"""
tests/test_data_processing.py
Unit tests for data_processing module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pytest
import pandas as pd
from modules.data_processing import (
    map_columns, clean_dataframe, parse_csv,
    get_monthly_summary, get_category_summary, _infer_transaction_type,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_content():
    return """date,description,amount,type,category,payment_method
2025-01-15,Swiggy order,350.00,expense,Food & Dining,UPI
2025-01-20,Monthly Salary,50000.00,income,Income,NEFT
2025-01-25,Amazon purchase,1200.00,expense,Shopping,Credit Card
2025-02-01,Monthly Salary,50000.00,income,Income,NEFT
2025-02-10,Uber ride,250.00,expense,Transportation,UPI
"""


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date":           ["2025-01-15","2025-01-20","2025-02-10","2025-02-20"],
        "description":    ["Swiggy","Salary","Uber","Amazon"],
        "amount":         [350.0, 50000.0, 250.0, 1200.0],
        "type":           ["expense","income","expense","expense"],
        "category":       ["Food & Dining","Income","Transportation","Shopping"],
        "payment_method": ["UPI","NEFT","UPI","Credit Card"],
    })


# ── Column Mapping ────────────────────────────────────────────────

def test_map_columns_standard():
    df = pd.DataFrame({"date":[],"description":[],"amount":[]})
    result = map_columns(df)
    assert "date" in result.columns
    assert "description" in result.columns
    assert "amount" in result.columns


def test_map_columns_aliases():
    df = pd.DataFrame({"Transaction_Date":[],"Narration":[],"amt":[]})
    result = map_columns(df)
    assert "date" in result.columns
    assert "description" in result.columns
    assert "amount" in result.columns


def test_map_columns_missing_raises():
    df = pd.DataFrame({"foo":["a"],"bar":["b"]})
    with pytest.raises(ValueError, match="Could not detect required column"):
        map_columns(df)


# ── Cleaning ──────────────────────────────────────────────────────

def test_clean_dataframe_basic(sample_df):
    cleaned, report = clean_dataframe(sample_df)
    assert len(cleaned) == 4
    assert report["final_rows"] == 4
    assert cleaned["amount"].dtype == float


def test_clean_removes_zero_amounts():
    df = pd.DataFrame({
        "date":["2025-01-01","2025-01-02"],
        "description":["Valid","Zero"],
        "amount":[100.0, 0.0],
    })
    cleaned, _ = clean_dataframe(df)
    assert len(cleaned) == 1


def test_clean_parses_rupee_amounts():
    df = pd.DataFrame({
        "date":["2025-01-01"],
        "description":["Test"],
        "amount":["₹1,500.00"],
    })
    cleaned, _ = clean_dataframe(df)
    assert cleaned["amount"].iloc[0] == 1500.0


def test_clean_removes_duplicates():
    df = pd.DataFrame({
        "date":["2025-01-01","2025-01-01"],
        "description":["Swiggy","Swiggy"],
        "amount":[350.0, 350.0],
    })
    cleaned, report = clean_dataframe(df)
    assert len(cleaned) == 1
    assert report["dropped_duplicate"] == 1


def test_clean_drops_invalid_dates_and_keeps_valid_rows():
    df = pd.DataFrame({
        "date":["2025-01-01", "not-a-date", "", None],
        "description":["Valid expense", "Broken entry", "Missing date", "Missing date"],
        "amount":[500.0, 200.0, 100.0, 50.0],
    })
    cleaned, _ = clean_dataframe(df)
    assert len(cleaned) == 1
    assert str(cleaned["date"].iloc[0]) == "2025-01-01 00:00:00"


def test_clean_infers_type_from_description():
    df = pd.DataFrame({
        "date":["2025-01-01","2025-01-02"],
        "description":["Monthly salary credit","Swiggy order"],
        "amount":[50000.0, 350.0],
    })
    cleaned, _ = clean_dataframe(df)
    assert cleaned[cleaned["description"].str.contains("salary")]["type"].iloc[0] == "income"


# ── CSV Parsing ───────────────────────────────────────────────────

def test_parse_csv_valid(sample_csv_content):
    file_obj = io.StringIO(sample_csv_content)
    df, report = parse_csv(file_obj)
    assert len(df) == 5
    assert report["final_rows"] == 5


def test_parse_csv_empty():
    file_obj = io.StringIO("date,description,amount\n")
    with pytest.raises(ValueError):
        parse_csv(file_obj)


# ── Aggregations ──────────────────────────────────────────────────

def test_monthly_summary(sample_df):
    sample_df["date"] = pd.to_datetime(sample_df["date"])
    monthly = get_monthly_summary(sample_df)
    assert len(monthly) >= 2
    assert "income" in monthly.columns
    assert "expense" in monthly.columns
    assert "savings" in monthly.columns


def test_category_summary(sample_df):
    sample_df["date"] = pd.to_datetime(sample_df["date"])
    cat = get_category_summary(sample_df)
    assert "category" in cat.columns
    assert "total" in cat.columns
    assert "pct" in cat.columns
    assert cat["pct"].sum() == pytest.approx(100.0, abs=0.5)


def test_category_summary_empty():
    df = pd.DataFrame(columns=["date","description","amount","type","category"])
    cat = get_category_summary(df)
    assert cat.empty
