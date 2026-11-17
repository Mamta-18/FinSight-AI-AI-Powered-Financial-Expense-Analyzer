"""
pages/reports.py
MODULE 7 – Report generation and CSV/text downloads.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.report_generation import (
    transactions_to_csv, generate_monthly_report_csv,
    generate_category_report_csv, generate_text_summary,
    get_spending_pattern_summary,
)


def show(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📄 Financial Reports</h1>
        <p class="page-subtitle">Download your data, generate summaries, and export reports.</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("📭 No data available. Upload transactions first.")
        return

    today   = datetime.now().strftime("%Y%m%d")
    summary = get_spending_pattern_summary(df)

    # ── Spending Pattern Summary Cards ─────────────────────────────
    st.markdown("#### 📊 Spending Pattern Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Expense Category", summary.get("top_category","—"),
              delta=f"{summary.get('top_category_pct',0):.1f}% of spend")
    c2.metric("Best Savings Month",   summary.get("best_month","—"))
    c3.metric("Highest Spend Month",  summary.get("worst_month","—"))

    c4, c5, c6 = st.columns(3)
    c4.metric("Total Income",        f"₹{summary.get('total_income',0):,.0f}")
    c5.metric("Total Expenses",      f"₹{summary.get('total_expense',0):,.0f}")
    c6.metric("Savings Rate",        f"{summary.get('savings_rate',0):.1f}%")

    st.markdown("---")

    # ── Download Section ───────────────────────────────────────────
    st.markdown("#### 📥 Download Reports")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                    border-radius:14px;padding:20px;height:180px;">
            <h4 style="color:#f1f5f9;margin:0 0 8px 0;">🧾 All Transactions CSV</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">
                Export all transactions with date, description, amount, category, and payment method.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Transactions CSV",
            data=transactions_to_csv(df),
            file_name=f"finsight_transactions_{today}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_b:
        st.markdown("""
        <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                    border-radius:14px;padding:20px;height:180px;">
            <h4 style="color:#f1f5f9;margin:0 0 8px 0;">📅 Monthly Summary CSV</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">
                Monthly income, expense, savings, and savings rate breakdown.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Monthly Report CSV",
            data=generate_monthly_report_csv(df),
            file_name=f"finsight_monthly_{today}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("""
        <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                    border-radius:14px;padding:20px;height:180px;">
            <h4 style="color:#f1f5f9;margin:0 0 8px 0;">🗂️ Category Report CSV</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">
                Category-wise expense totals, counts, and percentage breakdown.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Category Report CSV",
            data=generate_category_report_csv(df),
            file_name=f"finsight_categories_{today}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_d:
        st.markdown("""
        <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                    border-radius:14px;padding:20px;height:180px;">
            <h4 style="color:#f1f5f9;margin:0 0 8px 0;">📋 Full Summary Report (TXT)</h4>
            <p style="color:#94a3b8;font-size:0.85rem;">
                Human-readable financial summary with KPIs, top categories, and monthly breakdown.
            </p>
        </div>
        """, unsafe_allow_html=True)
        summary_text = generate_text_summary(df)
        st.download_button(
            label="⬇️ Download Summary Report TXT",
            data=summary_text.encode("utf-8"),
            file_name=f"finsight_summary_{today}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Preview Text Report ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 👁️ Preview: Text Summary Report")
    with st.expander("Show full report"):
        st.code(generate_text_summary(df), language=None)
