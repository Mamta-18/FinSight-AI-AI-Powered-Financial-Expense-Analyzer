"""
modules/financial_analytics.py
KPI calculations, trend analysis, and AI insights for FinSight AI.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# KPI CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute top-level KPI metrics from the full transaction DataFrame."""
    if df.empty:
        return _empty_kpis()

    income_df  = df[df["type"] == "income"]
    expense_df = df[df["type"] == "expense"]

    total_income  = income_df["amount"].sum()
    total_expense = expense_df["amount"].sum()
    total_savings = total_income - total_expense
    savings_rate  = (total_savings / total_income * 100) if total_income > 0 else 0.0

    # Monthly averages
    months = df["date"].dt.to_period("M").nunique()
    avg_monthly_income  = total_income  / months if months > 0 else 0
    avg_monthly_expense = total_expense / months if months > 0 else 0

    # MoM deltas (last full month vs second-to-last)
    income_delta, expense_delta, savings_delta = _mom_deltas(df)

    return {
        "total_income":         total_income,
        "total_expense":        total_expense,
        "total_savings":        total_savings,
        "savings_rate":         round(savings_rate, 1),
        "avg_monthly_income":   avg_monthly_income,
        "avg_monthly_expense":  avg_monthly_expense,
        "income_delta_pct":     income_delta,
        "expense_delta_pct":    expense_delta,
        "savings_delta_pct":    savings_delta,
        "transaction_count":    len(df),
        "months_covered":       months,
    }


def _empty_kpis() -> dict:
    return {k: 0.0 for k in [
        "total_income", "total_expense", "total_savings", "savings_rate",
        "avg_monthly_income", "avg_monthly_expense",
        "income_delta_pct", "expense_delta_pct", "savings_delta_pct",
        "transaction_count", "months_covered",
    ]}


def _mom_deltas(df: pd.DataFrame) -> tuple[float, float, float]:
    """Compute month-over-month change percentage for income, expense, savings."""
    df = df.copy()
    df["period"] = df["date"].dt.to_period("M")
    periods = sorted(df["period"].unique())

    if len(periods) < 2:
        return 0.0, 0.0, 0.0

    last   = periods[-1]
    second = periods[-2]

    def _sum(period, txn_type):
        return df[(df["period"] == period) & (df["type"] == txn_type)]["amount"].sum()

    for p in [last, second]:
        pass  # just ensure periods are iterable

    curr_inc  = _sum(last, "income");   prev_inc  = _sum(second, "income")
    curr_exp  = _sum(last, "expense");  prev_exp  = _sum(second, "expense")
    curr_sav  = curr_inc - curr_exp;    prev_sav  = prev_inc - prev_exp

    def _pct(curr, prev):
        if prev == 0:
            return 0.0
        return round((curr - prev) / abs(prev) * 100, 1)

    return _pct(curr_inc, prev_inc), _pct(curr_exp, prev_exp), _pct(curr_sav, prev_sav)


# ═══════════════════════════════════════════════════════════════════════════════
# MONTHLY AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_monthly_income_expense(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly income, expense, savings, and savings-rate data."""
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expense", "savings", "savings_rate", "month_str"])

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    income  = df[df["type"] == "income"].groupby("month")["amount"].sum()
    expense = df[df["type"] == "expense"].groupby("month")["amount"].sum()

    monthly = pd.DataFrame({"income": income, "expense": expense}).fillna(0)
    monthly["savings"] = monthly["income"] - monthly["expense"]
    monthly["savings_rate"] = np.where(
        monthly["income"] > 0,
        monthly["savings"] / monthly["income"] * 100,
        0.0,
    )
    monthly = monthly.reset_index()
    monthly["month_str"] = monthly["month"].dt.strftime("%b %Y")
    monthly["month"] = monthly["month"].astype(str)
    return monthly.sort_values("month")


def get_category_expense(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top N expense categories with totals and percentages."""
    exp = df[df["type"] == "expense"].copy()
    if exp.empty:
        return pd.DataFrame(columns=["category", "total", "pct"])

    total = exp["amount"].sum()
    summary = (
        exp.groupby("category")["amount"]
        .sum()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
        .head(top_n)
    )
    summary["pct"] = (summary["total"] / total * 100).round(1)
    return summary


def get_monthly_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return pivot table: months as rows, categories as columns, amounts as values."""
    exp = df[df["type"] == "expense"].copy()
    if exp.empty:
        return pd.DataFrame()

    exp["month"] = exp["date"].dt.to_period("M").astype(str)
    pivot = exp.pivot_table(
        index="month", columns="category", values="amount",
        aggfunc="sum", fill_value=0
    ).reset_index()
    return pivot


def get_top_transactions(df: pd.DataFrame, n: int = 10, txn_type: str = "expense") -> pd.DataFrame:
    """Return top N transactions by amount."""
    filtered = df[df["type"] == txn_type].copy()
    if filtered.empty:
        return pd.DataFrame()
    return filtered.nlargest(n, "amount")[["date", "description", "amount", "category", "payment_method"]]


def get_spending_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return rolling 3-month average expense trend."""
    monthly = get_monthly_income_expense(df)
    if len(monthly) < 2:
        return monthly
    monthly["rolling_expense"] = monthly["expense"].rolling(3, min_periods=1).mean()
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame) -> list[dict]:
    """
    Generate actionable text insights from transaction data.
    Each insight has: type, icon, text
    Only generates insights when data supports them.
    """
    insights: list[dict] = []
    if df.empty or len(df) < 5:
        return [{"type": "info", "icon": "ℹ️", "text": "Upload transactions to see personalized insights."}]

    kpis    = compute_kpis(df)
    monthly = get_monthly_income_expense(df)
    cat_exp = get_category_expense(df, top_n=3)

    # 1. Top expense category
    if not cat_exp.empty:
        top_cat  = cat_exp.iloc[0]["category"]
        top_pct  = cat_exp.iloc[0]["pct"]
        insights.append({
            "type": "warning" if top_pct > 30 else "info",
            "icon": "🍽️" if "Food" in top_cat else "🛍️",
            "text": f"**{top_cat}** is your highest expense category at **{top_pct}%** of total spending.",
        })

    # 2. Savings rate assessment
    sr = kpis["savings_rate"]
    if sr >= 20:
        insights.append({"type": "success", "icon": "✅", "text": f"Great job! You maintained a healthy savings rate of **{sr}%**."})
    elif sr > 0:
        insights.append({"type": "warning", "icon": "⚠️", "text": f"Your savings rate is **{sr}%**. Financial experts recommend saving at least 20%."})
    else:
        insights.append({"type": "danger", "icon": "🚨", "text": f"Your expenses exceed your income. Savings rate is **{sr}%**. Review your spending urgently."})

    # 3. Month-over-month expense trend
    if len(monthly) >= 2:
        last_exp   = monthly.iloc[-1]["expense"]
        prev_exp   = monthly.iloc[-2]["expense"]
        if prev_exp > 0:
            change_pct = (last_exp - prev_exp) / prev_exp * 100
            last_month = monthly.iloc[-1]["month_str"]
            if change_pct > 10:
                insights.append({"type": "warning", "icon": "📈",
                    "text": f"Your expenses **increased by {change_pct:.1f}%** last month ({last_month}). Consider reviewing discretionary spending."})
            elif change_pct < -10:
                insights.append({"type": "success", "icon": "📉",
                    "text": f"Your expenses **decreased by {abs(change_pct):.1f}%** last month ({last_month}). Keep it up!"})
            else:
                insights.append({"type": "info", "icon": "📊",
                    "text": f"Your expenses are **stable** (~{change_pct:+.1f}% change) compared to the previous month."})

    # 4. Average monthly savings
    avg_sav = kpis["avg_monthly_income"] - kpis["avg_monthly_expense"]
    if avg_sav > 0:
        insights.append({"type": "success", "icon": "💰",
            "text": f"On average, you save **₹{avg_sav:,.0f}** per month."})

    # 5. Category diversity
    unique_cats = df[df["type"] == "expense"]["category"].nunique()
    if unique_cats >= 6:
        insights.append({"type": "info", "icon": "🗂️",
            "text": f"Your expenses are spread across **{unique_cats} categories**, indicating diverse spending habits."})

    return insights


def generate_budget_suggestions(df: pd.DataFrame) -> list[str]:
    """Generate specific budget reduction suggestions based on data."""
    suggestions: list[str] = []
    if df.empty:
        return suggestions

    cat_exp = get_category_expense(df)
    kpis    = compute_kpis(df)
    monthly = get_monthly_income_expense(df)

    if cat_exp.empty:
        return suggestions

    for _, row in cat_exp.iterrows():
        cat  = row["category"]
        pct  = row["pct"]
        amt  = row["total"]
        months = kpis["months_covered"] or 1
        monthly_avg = amt / months

        if cat == "Food & Dining" and pct > 20:
            suggestions.append(f"🍽️ Reduce dining out — try cooking at home 3x/week to save ≈₹{monthly_avg*0.3:,.0f}/month.")
        elif cat == "Entertainment" and pct > 10:
            suggestions.append(f"🎬 Review streaming subscriptions — cancel unused ones to save ≈₹{monthly_avg*0.4:,.0f}/month.")
        elif cat == "Shopping" and pct > 20:
            suggestions.append(f"🛍️ Set a monthly shopping budget of ₹{monthly_avg*0.7:,.0f} to reduce impulse spending.")
        elif cat == "Transportation" and pct > 15:
            suggestions.append(f"🚗 Consider carpooling or public transport to cut transport costs by ≈20%.")

    if kpis["savings_rate"] < 20:
        target = kpis["avg_monthly_income"] * 0.2
        suggestions.append(f"💡 Set up an automatic transfer of ₹{target:,.0f}/month to your savings account.")

    return suggestions[:5]
