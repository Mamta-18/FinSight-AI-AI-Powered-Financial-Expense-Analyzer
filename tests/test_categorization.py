"""
tests/test_categorization.py
Unit tests for expense_categorization module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from modules.expense_categorization import (
    rule_based_categorize, categorize_dataframe,
    _preprocess_text, train_ml_model, get_model_status,
)


# ── Text Preprocessing ────────────────────────────────────────────

def test_preprocess_text_lowercases():
    assert _preprocess_text("SWIGGY ORDER") == "swiggy order"


def test_preprocess_text_removes_punctuation():
    result = _preprocess_text("Uber, Ride!")
    assert "," not in result
    assert "!" not in result


def test_preprocess_text_normalizes_spaces():
    result = _preprocess_text("swiggy   order")
    assert "  " not in result


# ── Rule-Based Categorizer ────────────────────────────────────────

@pytest.mark.parametrize("description,expected_cat", [
    ("Swiggy order biryani",      "Food & Dining"),
    ("Zomato pizza delivery",     "Food & Dining"),
    ("Netflix subscription",      "Entertainment"),
    ("Uber ride to office",       "Transportation"),
    ("Electricity bill BESCOM",   "Bills & Utilities"),
    ("Amazon purchase shoes",     "Shopping"),
    ("BigBasket grocery order",   "Groceries"),
    ("Apollo Pharmacy medicines", "Healthcare"),
    ("IndiGo flight ticket",      "Travel"),
    ("Udemy Python course",       "Education"),
    ("Monthly salary credit",     "Income"),
])
def test_rule_based_known_categories(description, expected_cat):
    cat, conf = rule_based_categorize(description)
    assert cat == expected_cat, f"Expected '{expected_cat}' for '{description}', got '{cat}'"
    assert 0.0 <= conf <= 1.0


def test_rule_based_unknown_returns_other():
    cat, conf = rule_based_categorize("xyxy random gibberish qwerty123")
    assert cat == "Other"
    assert conf < 0.6


def test_rule_based_confidence_range():
    cat, conf = rule_based_categorize("Swiggy zomato food restaurant")
    assert 0.0 <= conf <= 1.0


# ── DataFrame Categorization ──────────────────────────────────────

@pytest.fixture
def expense_df():
    return pd.DataFrame({
        "description": [
            "Swiggy order", "Netflix sub", "Uber ride",
            "Amazon purchase", "Electricity bill",
        ],
        "amount":   [350, 199, 150, 2000, 1200],
        "type":     ["expense"] * 5,
        "category": ["Other"] * 5,
    })


def test_categorize_dataframe_adds_columns(expense_df):
    result = categorize_dataframe(expense_df, use_ml=False)
    assert "predicted_category" in result.columns
    assert "confidence" in result.columns
    assert "method" in result.columns


def test_categorize_dataframe_rule_based(expense_df):
    result = categorize_dataframe(expense_df, use_ml=False)
    assert all(result["method"] == "rule-based")


def test_categorize_dataframe_swiggy(expense_df):
    result = categorize_dataframe(expense_df, use_ml=False)
    swiggy_row = result[result["description"] == "Swiggy order"]
    assert swiggy_row["predicted_category"].iloc[0] == "Food & Dining"


def test_categorize_dataframe_confidence_valid(expense_df):
    result = categorize_dataframe(expense_df, use_ml=False)
    assert all(0.0 <= c <= 1.0 for c in result["confidence"])


# ── ML Training ───────────────────────────────────────────────────

def test_train_ml_model_insufficient_data():
    df = pd.DataFrame({
        "description": ["Swiggy"] * 10,
        "type":        ["expense"] * 10,
        "category":    ["Food & Dining"] * 10,
    })
    result = train_ml_model(df)
    assert result["status"] == "insufficient_data"


def test_train_ml_model_sufficient_data():
    """Create 60 labeled examples across 3 categories."""
    descriptions = (
        ["Swiggy order biryani"] * 20 +
        ["Netflix subscription"] * 20 +
        ["Uber ride to airport"] * 20
    )
    categories = (
        ["Food & Dining"] * 20 +
        ["Entertainment"] * 20 +
        ["Transportation"] * 20
    )
    df = pd.DataFrame({
        "description": descriptions,
        "type":        ["expense"] * 60,
        "category":    categories,
    })
    result = train_ml_model(df)
    assert result["status"] == "success"
    assert result["samples"] == 60
    assert result["cv_accuracy"] >= 0  # just verify it ran


# ── Model Status ──────────────────────────────────────────────────

def test_get_model_status_returns_dict():
    status = get_model_status()
    assert isinstance(status, dict)
    assert "available" in status
