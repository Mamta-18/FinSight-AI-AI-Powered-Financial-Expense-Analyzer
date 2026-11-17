"""
modules/report_generation.py
CSV export, summary report generation, and downloadable content.
"""

import io
from datetime import datetime
import pandas as pd
from modules.financial_analytics import (
    compute_kpis, get_category_expense,
    get_monthly_income_expense, get_top_transactions,
)


def transactions_to_csv(df: pd.DataFrame) -> bytes:
    """Convert transactions DataFrame to CSV bytes for download."""
    export_cols = ["transaction_id", "date", "description", "amount", "type", "category", "payment_method"]
    available   = [c for c in export_cols if c in df.columns]
    df_out = df[available].copy()
    if "date" in df_out.columns:
        df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
    return df_out.to_csv(index=False).encode("utf-8")


def generate_monthly_report_csv(df: pd.DataFrame) -> bytes:
    """Generate a monthly income/expense/savings summary CSV."""
    monthly = get_monthly_income_expense(df)
    return monthly.to_csv(index=False).encode("utf-8")


def generate_category_report_csv(df: pd.DataFrame) -> bytes:
    """Generate a category-wise expense summary CSV."""
    cat_df = get_category_expense(df, top_n=20)
    return cat_df.to_csv(index=False).encode("utf-8")


def generate_text_summary(df: pd.DataFrame) -> str:
    """Generate a readable financial summary report as plain text."""
    if df.empty:
        return "No transaction data available to generate a report."

    kpis    = compute_kpis(df)
    cat_exp = get_category_expense(df, top_n=5)
    monthly = get_monthly_income_expense(df)
    top_txn = get_top_transactions(df, n=5)

    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    start = df["date"].min().strftime("%d %b %Y")
    end   = df["date"].max().strftime("%d %b %Y")

    lines = [
        "=" * 60,
        "         FINSIGHT AI – FINANCIAL SUMMARY REPORT",
        "=" * 60,
        f"  Generated: {now}",
        f"  Period:    {start} – {end}",
        "=" * 60,
        "",
        "KEY METRICS",
        "-" * 40,
        f"  Total Income:        ₹{kpis['total_income']:>12,.0f}",
        f"  Total Expenses:      ₹{kpis['total_expense']:>12,.0f}",
        f"  Total Savings:       ₹{kpis['total_savings']:>12,.0f}",
        f"  Savings Rate:        {kpis['savings_rate']:>12.1f}%",
        f"  Avg Monthly Expense: ₹{kpis['avg_monthly_expense']:>12,.0f}",
        f"  Months Covered:      {kpis['months_covered']:>12}",
        f"  Transactions:        {kpis['transaction_count']:>12}",
        "",
        "EXPENSE BY CATEGORY (Top 5)",
        "-" * 40,
    ]

    for _, row in cat_exp.iterrows():
        bar = "█" * int(row["pct"] / 2)
        lines.append(f"  {row['category']:<20} ₹{row['total']:>10,.0f}  ({row['pct']:>5.1f}%)  {bar}")

    lines += [
        "",
        "MONTHLY BREAKDOWN (Last 6 Months)",
        "-" * 40,
        f"  {'Month':<12} {'Income':>12} {'Expense':>12} {'Savings':>12}",
    ]

    for _, row in monthly.tail(6).iterrows():
        lines.append(
            f"  {row['month_str']:<12} ₹{row['income']:>10,.0f}  ₹{row['expense']:>10,.0f}  ₹{row['savings']:>10,.0f}"
        )

    if not top_txn.empty:
        lines += [
            "",
            "TOP 5 EXPENSES",
            "-" * 40,
        ]
        for _, row in top_txn.iterrows():
            date_str = row["date"].strftime("%d %b") if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            lines.append(
                f"  {date_str}  {str(row['description'])[:30]:<30}  ₹{row['amount']:>10,.0f}  {row['category']}"
            )

    lines += [
        "",
        "=" * 60,
        "  ⚠️  This report is for personal financial tracking only.",
        "  FinSight AI is not a certified financial advisor.",
        "=" * 60,
    ]

    return "\n".join(lines)


def get_spending_pattern_summary(df: pd.DataFrame) -> dict:
    """Return a structured dict summarizing spending patterns."""
    if df.empty:
        return {}

    kpis    = compute_kpis(df)
    cat_exp = get_category_expense(df)
    monthly = get_monthly_income_expense(df)

    top_cat = cat_exp.iloc[0]["category"] if not cat_exp.empty else "N/A"
    top_pct = cat_exp.iloc[0]["pct"]      if not cat_exp.empty else 0

    # Month with highest expense
    if not monthly.empty:
        worst_month_idx = monthly["expense"].idxmax()
        best_month_idx  = monthly["savings"].idxmax()
        worst_month = monthly.loc[worst_month_idx, "month_str"]
        best_month  = monthly.loc[best_month_idx,  "month_str"]
    else:
        worst_month = best_month = "N/A"

    return {
        "total_income":   kpis["total_income"],
        "total_expense":  kpis["total_expense"],
        "savings_rate":   kpis["savings_rate"],
        "top_category":   top_cat,
        "top_category_pct": top_pct,
        "worst_month":    worst_month,
        "best_month":     best_month,
        "avg_monthly_expense": kpis["avg_monthly_expense"],
    }
