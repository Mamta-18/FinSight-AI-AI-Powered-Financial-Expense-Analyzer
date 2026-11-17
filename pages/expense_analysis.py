"""
pages/expense_analysis.py
MODULE 4 – Detailed financial analytics with deep-dive charts and insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from modules.financial_analytics import (
    compute_kpis, get_category_expense,
    get_monthly_income_expense, get_monthly_category_breakdown,
    get_top_transactions, generate_budget_suggestions,
)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

COLORS = ["#3b82f6","#8b5cf6","#06b6d4","#f97316","#10b981","#f43f5e","#84cc16","#0ea5e9","#a855f7","#6b7280"]


def show(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📈 Expense Analysis</h1>
        <p class="page-subtitle">Deep-dive into your spending patterns, category breakdowns, and financial health.</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("📭 No data available. Please upload transactions first.")
        return

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Category Analysis",
        "📅 Monthly Comparison",
        "🏆 Top Transactions",
        "💡 Budget Suggestions",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 – Category Analysis
    # ════════════════════════════════════════════════════════════════
    with tab1:
        kpis    = compute_kpis(df)
        cat_exp = get_category_expense(df, top_n=10)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Expenses",    f"₹{kpis['total_expense']:,.0f}")
        c2.metric("Avg Monthly",       f"₹{kpis['avg_monthly_expense']:,.0f}")
        c3.metric("Categories Used",   str(df[df["type"]=="expense"]["category"].nunique()))

        st.markdown("---")
        col_bar, col_table = st.columns([6, 4])

        with col_bar:
            if not cat_exp.empty:
                fig = go.Figure(go.Bar(
                    x=cat_exp["total"],
                    y=cat_exp["category"],
                    orientation="h",
                    marker=dict(
                        color=cat_exp["pct"],
                        colorscale=[[0,"#3b82f6"],[0.5,"#8b5cf6"],[1,"#f43f5e"]],
                        showscale=False,
                    ),
                    text=[f"₹{v:,.0f} ({p}%)" for v, p in zip(cat_exp["total"], cat_exp["pct"])],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=11),
                    hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
                ))
                fig.update_layout(
                    **CHART_LAYOUT,
                    height=380,
                    title="Expenses by Category",
                    title_font=dict(color="#f1f5f9", size=14),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                    xaxis=dict(tickprefix="₹", tickformat=",.0f", gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("#### 📋 Category Summary")
            if not cat_exp.empty:
                for _, row in cat_exp.iterrows():
                    pct_bar_width = int(row["pct"])
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:3px;">
                            <span style="color:#f1f5f9;font-weight:600;">{row['category']}</span>
                            <span style="color:#94a3b8;">₹{row['total']:,.0f} ({row['pct']}%)</span>
                        </div>
                        <div style="background:rgba(59,130,246,0.15);border-radius:4px;height:6px;">
                            <div style="background:linear-gradient(90deg,#3b82f6,#8b5cf6);
                                        border-radius:4px;height:6px;width:{pct_bar_width}%;
                                        max-width:100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 – Monthly Comparison
    # ════════════════════════════════════════════════════════════════
    with tab2:
        monthly = get_monthly_income_expense(df)

        if monthly.empty:
            st.info("No monthly data available.")
        else:
            # Stacked area chart
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                name="Income", x=monthly["month_str"], y=monthly["income"],
                fill="tonexty", mode="lines",
                line=dict(color="#06b6d4", width=2),
                fillcolor="rgba(6,182,212,0.1)",
            ))
            fig_area.add_trace(go.Scatter(
                name="Expense", x=monthly["month_str"], y=monthly["expense"],
                fill="tozeroy", mode="lines",
                line=dict(color="#8b5cf6", width=2),
                fillcolor="rgba(139,92,246,0.15)",
            ))
            fig_area.add_trace(go.Scatter(
                name="Savings", x=monthly["month_str"], y=monthly["savings"],
                mode="lines+markers",
                line=dict(color="#10b981", width=2, dash="dot"),
                marker=dict(size=6, color="#10b981"),
            ))
            fig_area.update_layout(
                **CHART_LAYOUT,
                height=350,
                title="Income vs Expenses vs Savings (Monthly)",
                title_font=dict(color="#f1f5f9", size=14),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(tickprefix="₹", tickformat=",.0f", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_area, use_container_width=True)

            # Monthly table
            st.markdown("#### 📅 Monthly Breakdown Table")
            monthly_disp = monthly[["month_str","income","expense","savings","savings_rate"]].copy()
            monthly_disp.columns = ["Month","Income (₹)","Expense (₹)","Savings (₹)","Savings Rate (%)"]
            monthly_disp[["Income (₹)","Expense (₹)","Savings (₹)"]] = monthly_disp[
                ["Income (₹)","Expense (₹)","Savings (₹)"]
            ].map(lambda x: f"₹{x:,.0f}")
            monthly_disp["Savings Rate (%)"] = monthly_disp["Savings Rate (%)"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(monthly_disp, use_container_width=True)

            # Category heatmap by month
            st.markdown("#### 🔥 Spending Heatmap by Category & Month")
            pivot = get_monthly_category_breakdown(df)
            if not pivot.empty and len(pivot) > 1:
                cat_cols = [c for c in pivot.columns if c != "month"]
                fig_heat = go.Figure(go.Heatmap(
                    z=pivot[cat_cols].values.T,
                    x=pivot["month"].tolist(),
                    y=cat_cols,
                    colorscale=[[0,"#0d1b2e"],[0.5,"#3b82f6"],[1,"#f43f5e"]],
                    hovertemplate="<b>%{y}</b> | %{x}<br>₹%{z:,.0f}<extra></extra>",
                    text=[[f"₹{v:,.0f}" if v > 0 else "" for v in row] for row in pivot[cat_cols].values.T],
                    texttemplate="%{text}",
                    textfont=dict(size=9, color="white"),
                ))
                fig_heat.update_layout(
                    **CHART_LAYOUT,
                    height=350,
                    xaxis=dict(tickfont=dict(size=10)),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3 – Top Transactions
    # ════════════════════════════════════════════════════════════════
    with tab3:
        c_exp, c_inc = st.columns(2)
        with c_exp:
            st.markdown("#### 💸 Top 10 Expenses")
            top_exp = get_top_transactions(df, n=10, txn_type="expense")
            if not top_exp.empty:
                top_exp_disp = top_exp.copy()
                top_exp_disp["date"]   = pd.to_datetime(top_exp_disp["date"]).dt.strftime("%d %b")
                top_exp_disp["amount"] = top_exp_disp["amount"].apply(lambda x: f"₹{x:,.0f}")
                top_exp_disp["description"] = top_exp_disp["description"].str[:35]
                st.dataframe(top_exp_disp, use_container_width=True)
            else:
                st.info("No expense data.")

        with c_inc:
            st.markdown("#### 💰 Top 10 Income")
            top_inc = get_top_transactions(df, n=10, txn_type="income")
            if not top_inc.empty:
                top_inc_disp = top_inc.copy()
                top_inc_disp["date"]   = pd.to_datetime(top_inc_disp["date"]).dt.strftime("%d %b")
                top_inc_disp["amount"] = top_inc_disp["amount"].apply(lambda x: f"₹{x:,.0f}")
                top_inc_disp["description"] = top_inc_disp["description"].str[:35]
                st.dataframe(top_inc_disp, use_container_width=True)
            else:
                st.info("No income data.")

        # Payment method breakdown
        st.markdown("#### 💳 Payment Method Distribution")
        pay_counts = df[df["type"]=="expense"]["payment_method"].value_counts()
        if not pay_counts.empty:
            fig_pay = go.Figure(go.Pie(
                labels=pay_counts.index, values=pay_counts.values,
                hole=0.4,
                marker_colors=COLORS[:len(pay_counts)],
                textinfo="label+percent",
                textfont=dict(color="white", size=11),
            ))
            fig_pay.update_layout(**CHART_LAYOUT, height=320, showlegend=True)
            st.plotly_chart(fig_pay, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 – Budget Suggestions
    # ════════════════════════════════════════════════════════════════
    with tab4:
        suggestions = generate_budget_suggestions(df)
        kpis = compute_kpis(df)

        # Savings dial
        sr = kpis["savings_rate"]
        gauge_color = "#10b981" if sr >= 20 else "#f59e0b" if sr >= 10 else "#f43f5e"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sr,
            delta={"reference": 20, "suffix": "%"},
            title={"text": "Savings Rate", "font": {"color": "#f1f5f9", "size": 16}},
            number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 36}},
            gauge={
                "axis": {"range": [0, 60], "tickcolor": "#64748b"},
                "bar":  {"color": gauge_color},
                "steps": [
                    {"range": [0, 10],  "color": "rgba(244,63,94,0.15)"},
                    {"range": [10, 20], "color": "rgba(245,158,11,0.15)"},
                    {"range": [20, 60], "color": "rgba(16,185,129,0.15)"},
                ],
                "threshold": {"line": {"color": "#3b82f6", "width": 2}, "thickness": 0.75, "value": 20},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(59,130,246,0.3)",
            },
        ))
        fig_gauge.update_layout(**CHART_LAYOUT, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 💡 Personalised Budget Suggestions")
        if suggestions:
            for s in suggestions:
                st.markdown(f"""
                <div class="insight-card insight-info" style="font-size:0.9rem;">
                    {s}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card insight-success">
                ✅ Great financial health! No major budget concerns detected.
            </div>
            """, unsafe_allow_html=True)
