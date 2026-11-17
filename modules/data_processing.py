"""
modules/data_processing.py
CSV parsing, cleaning, validation, and column-mapping for FinSight AI.
"""

import uuid
import re
from typing import Optional
import pandas as pd
import numpy as np
from io import StringIO


REQUIRED_COLUMNS_ALIASES = {
    "date":           ["date", "transaction_date", "txn_date", "trans_date", "posted_date", "value_date"],
    "description":    ["description", "desc", "narration", "particulars", "remarks", "details", "merchant", "note"],
    "amount":         ["amount", "amt", "value", "debit_credit", "transaction_amount"],
    "type":           ["type", "transaction_type", "txn_type", "trans_type", "credit_debit"],
    "category":       ["category", "cat", "expense_category", "spending_category"],
    "payment_method": ["payment_method", "mode", "payment_mode", "method", "channel"],
}

INCOME_KEYWORDS = [
    "salary", "wages", "income", "credit", "deposit", "refund",
    "transfer in", "received", "reimbursement", "bonus", "dividend",
    "interest credited", "cashback"
]


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name from candidates (case-insensitive)."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for alias in candidates:
        if alias in lower_cols:
            return lower_cols[alias]
    return None


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detect and rename DataFrame columns to standard names.
    Returns a DataFrame with standard column names, dropping unmapped extras.
    """
    rename_map: dict[str, str] = {}
    for std_name, aliases in REQUIRED_COLUMNS_ALIASES.items():
        found = _detect_column(df, aliases)
        if found and found not in rename_map.values():
            rename_map[found] = std_name

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    for col in ["date", "description", "amount"]:
        if col not in df.columns:
            raise ValueError(
                f"Could not detect required column '{col}'. "
                f"Available columns: {list(df.columns)}"
            )
    return df


def _infer_transaction_type(row: pd.Series) -> str:
    """Infer income vs expense from amount sign or description."""
    if "type" in row and pd.notna(row["type"]):
        t = str(row["type"]).lower()
        if any(k in t for k in ["credit", "income", "cr"]):
            return "income"
        if any(k in t for k in ["debit", "expense", "dr"]):
            return "expense"

    # Positive = income if description suggests it
    desc = str(row.get("description", "")).lower()
    if any(k in desc for k in INCOME_KEYWORDS):
        return "income"

    # Negative amount → expense (some bank formats)
    try:
        amt = float(row.get("amount", 0))
        if amt < 0:
            return "expense"
        # large round amounts often salary
        if amt >= 20000 and any(k in desc for k in ["salary", "wages"]):
            return "income"
    except (ValueError, TypeError):
        pass

    return "expense"


def _normalize_amount_value(value) -> str:
    """Normalize common financial amount formats to a numeric string."""
    if pd.isna(value):
        return ""

    cleaned = str(value).strip()
    if cleaned.lower() in {"", "nan", "none", "null"}:
        return ""

    cleaned = cleaned.replace("₹", "").replace("$", "").replace(",", "")
    cleaned = re.sub(r"\((.*)\)", r"-\1", cleaned)
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    return cleaned


def _coerce_dates(series: pd.Series) -> pd.Series:
    """Convert date-like values to datetime while cleaning missing/invalid values."""
    if series.empty:
        return pd.to_datetime(series, errors="coerce")

    cleaned = series.copy()
    if cleaned.dtype == object:
        cleaned = cleaned.map(
            lambda value: pd.NaT
            if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}
            else value
        )

    return pd.to_datetime(cleaned, errors="coerce", format="mixed")


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and validate a mapped DataFrame.
    Returns: (cleaned_df, report_dict)
    """
    report = {
        "original_rows": len(df),
        "dropped_null": 0,
        "dropped_duplicate": 0,
        "dropped_invalid_amount": 0,
        "final_rows": 0,
    }

    # Drop fully empty rows
    before = len(df)
    df = df.dropna(how="all")
    report["dropped_null"] = before - len(df)

    # Parse dates while handling missing/invalid entries safely
    df["date"] = _coerce_dates(df["date"])
    before = len(df)
    df = df.dropna(subset=["date"]).copy()
    report["dropped_null"] += before - len(df)

    # Clean amount
    df["amount"] = df["amount"].map(_normalize_amount_value)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["amount"])
    df = df[df["amount"].abs() > 0]
    report["dropped_invalid_amount"] = before - len(df)

    # Use absolute value for amount
    df["amount"] = df["amount"].abs()

    # Description
    df["description"] = df["description"].fillna("Unknown").astype(str).str.strip()

    # Infer type if missing
    if "type" not in df.columns:
        df["type"] = df.apply(_infer_transaction_type, axis=1)
    else:
        df["type"] = df.apply(_infer_transaction_type, axis=1)

    # Category
    if "category" not in df.columns:
        df["category"] = "Other"
    df["category"] = df["category"].fillna("Other").astype(str).str.strip()

    # Payment method
    if "payment_method" not in df.columns:
        df["payment_method"] = "Unknown"
    df["payment_method"] = df["payment_method"].fillna("Unknown").astype(str)

    # Drop duplicates (same date + description + amount)
    before = len(df)
    df = df.drop_duplicates(subset=["date", "description", "amount"])
    report["dropped_duplicate"] = before - len(df)

    # Generate transaction IDs
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [
            str(uuid.uuid4())[:8].upper() for _ in range(len(df))
        ]

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    report["final_rows"] = len(df)
    return df, report


def parse_csv(file_obj) -> tuple[pd.DataFrame, dict]:
    """
    Parse a CSV file object (Streamlit UploadedFile or file path).
    Returns: (cleaned_df, report)
    Raises: ValueError if the file cannot be parsed.
    """
    try:
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            df = pd.read_csv(StringIO(content))
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}") from e

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    df = map_columns(df)
    df, report = clean_dataframe(df)
    return df, report


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Return a list of validation warnings for a cleaned DataFrame."""
    warnings_list = []
    if len(df) < 10:
        warnings_list.append("⚠️ Less than 10 transactions found. Analytics may be limited.")
    if df["date"].dt.year.nunique() < 1:
        warnings_list.append("⚠️ Could not detect date range reliably.")
    if (df["type"] == "income").sum() == 0:
        warnings_list.append("⚠️ No income transactions detected. Savings calculations will be zero.")
    return warnings_list


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly income, expense, and savings summary."""
    if df.empty:
        return pd.DataFrame(columns=["year_month", "income", "expense", "savings", "savings_rate"])

    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    income_df = df[df["type"] == "income"].groupby("year_month")["amount"].sum().rename("income")
    expense_df = df[df["type"] == "expense"].groupby("year_month")["amount"].sum().rename("expense")

    monthly = pd.concat([income_df, expense_df], axis=1).fillna(0)
    monthly["savings"] = monthly["income"] - monthly["expense"]
    monthly["savings_rate"] = monthly.apply(
        lambda r: (r["savings"] / r["income"] * 100) if r["income"] > 0 else 0, axis=1
    )
    monthly = monthly.reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return category-wise expense summary."""
    expenses = df[df["type"] == "expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=["category", "total", "count", "pct"])

    total = expenses["amount"].sum()
    summary = (
        expenses.groupby("category")["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total", "count": "count"})
        .reset_index()
    )
    summary["pct"] = (summary["total"] / total * 100).round(1)
    return summary.sort_values("total", ascending=False)
