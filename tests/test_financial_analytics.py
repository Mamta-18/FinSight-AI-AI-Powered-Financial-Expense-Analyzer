"""Regression tests for financial analytics aggregations."""

import pandas as pd
import pytest

from modules.financial_analytics import get_monthly_income_expense


def test_monthly_income_expense_includes_savings_rate():
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-05", "2025-01-10", "2025-02-05"]),
            "type": ["income", "expense", "expense"],
            "amount": [1000.0, 250.0, 100.0],
        }
    )

    monthly = get_monthly_income_expense(transactions)

    assert "savings_rate" in monthly.columns
    assert monthly.loc[0, "savings_rate"] == pytest.approx(75.0)
    assert monthly.loc[1, "savings_rate"] == pytest.approx(0.0)


def test_monthly_income_expense_empty_has_expected_columns():
    transactions = pd.DataFrame(columns=["date", "type", "amount"])

    monthly = get_monthly_income_expense(transactions)

    assert "savings_rate" in monthly.columns
