"""
pages/dashboard.py
MODULE 1 – Dashboard Overview with KPI cards and 4 Plotly charts.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from modules.financial_analytics import (
    compute_kpis, get_monthly_income_expense,
    get_category_expense, get_spending_trends, generate_insights,
)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)

CATEGORY_COLORS = {
    "Food & Dining":    "#f97316",
    "Groceries":        "#84cc16",
    "Shopping":         "#a855f7",
    "Transportation":   "#06b6d4",
    "Bills & Utilities":"#3b82f6",
    "Healthcare":       "#10b981",
    "Entertainment":    "#f43f5e",
    "Education":        "#8b5cf6",
    "Travel":           "#0ea5e9",
    "Other":            "#6b7280",
}


def _fmt(val: float) -> str:
    if val >= 1_00_000:
        return f"₹{val/1_00_000:.1f}L"
    if val >= 1_000:
        return f"₹{val:,.0f}"
    return f"₹{val:.0f}"


def _kpi_card(label: str, value: float, delta: float, icon: str, card_class: str):
    sign   = "+" if delta >= 0 else ""
    color  = "#10b981" if delta >= 0 else "#f43f5e"
    arrow  = "↑" if delta >= 0 else "↓"
    st.markdown(f"""
    <div class="kpi-card {card_class}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{_fmt(value)}</div>
        <div style="color:{color};font-size:0.8rem;font-weight:600;">
            {arrow} {sign}{delta:.1f}% vs last month
        </div>
    </div>
    """, unsafe_allow_html=True)


def show(df: pd.DataFrame):
    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-panel">
        <div class="hero-badge">✦ FinSight AI</div>
        <h1 class="hero-title">AI-Powered<br><span class="gradient">Financial Expense</span><br>Analyzer</h1>
        <p class="hero-subtitle">Analyze • Predict • Get AI Insights • Chat with AI</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:rgba(15,35,64,0.5);
                    border-radius:16px;border:1px dashed rgba(59,130,246,0.3);">
            <div style="font-size:4rem;margin-bottom:16px;">📊</div>
            <h3 style="color:#f1f5f9;">No Data Yet</h3>
            <p style="color:#64748b;">Upload your transaction CSV on the <b>Transactions</b> page
            or load the demo dataset from <b>Settings</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    kpis    = compute_kpis(df)
    monthly = get_monthly_income_expense(df)
    cat_exp = get_category_expense(df, top_n=10)
    trends  = get_spending_trends(df)
    insights = generate_insights(df)

    # ── Date Range Filter ──────────────────────────────────────────────
    col_fl, col_fr = st.columns([3, 1])
    with col_fl:
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        date_range = st.date_input(
            "📅 Date Range",
            value=(min_d, max_d),
            min_value=min_d, max_value=max_d,
            label_visibility="collapsed",
        )
    with col_fr:
        all_cats = ["All"] + sorted(df[df["type"] == "expense"]["category"].unique().tolist())
        cat_filter = st.selectbox("Category", all_cats, label_visibility="collapsed")

    # Apply filters
    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[
            (filtered["date"].dt.date >= date_range[0]) &
            (filtered["date"].dt.date <= date_range[1])
        ]
    if cat_filter != "All":
        # For KPIs keep all types visible but highlight the category
        pass

    # Recompute KPIs on filtered data
    fkpis = compute_kpis(filtered)

    # ── KPI Cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Total Income",   fkpis["total_income"],  fkpis["income_delta_pct"],  "💳", "kpi-card-income")
    with c2:
        _kpi_card("Total Expenses", fkpis["total_expense"], fkpis["expense_delta_pct"], "🛒", "kpi-card-expense")
    with c3:
        _kpi_card("Total Savings",  fkpis["total_savings"], fkpis["savings_delta_pct"], "🐷", "kpi-card-savings")
    with c4:
        # Savings rate card (special render)
        sr = fkpis["savings_rate"]
        st.markdown(f"""
        <div class="kpi-card kpi-card-rate">
            <div class="kpi-icon">%</div>
            <div class="kpi-label">Savings Rate</div>
            <div class="kpi-value">{sr:.1f}%</div>
            <div style="color:#8b5cf6;font-size:0.8rem;font-weight:600;">
                {'✅ Healthy' if sr >= 20 else '⚠️ Below target (20%)'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Row 2: Category Donut + Monthly Bar ───────────────────────────
    col_donut, col_bar = st.columns([4, 6])

    with col_donut:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>📊 Expense by Category</div>", unsafe_allow_html=True)

        f_cat = get_category_expense(filtered)
        if not f_cat.empty:
            colors = [CATEGORY_COLORS.get(c, "#6366f1") for c in f_cat["category"]]
            fig_donut = go.Figure(go.Pie(
                labels=f_cat["category"],
                values=f_cat["total"],
                hole=0.6,
                marker_colors=colors,
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            total_exp = f_cat["total"].sum()
            fig_donut.update_layout(
                **CHART_LAYOUT,
                height=300,
                annotations=[dict(
                    text=f"<b>₹{total_exp:,.0f}</b><br>Total",
                    x=0.5, y=0.5, font_size=14, font_color="#f1f5f9",
                    showarrow=False,
                )],
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Legend
            for _, row in f_cat.head(6).iterrows():
                col = CATEGORY_COLORS.get(row["category"], "#6b7280")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:3px 0;font-size:0.78rem;">'
                    f'<span><span style="color:{col}">●</span> {row["category"]}</span>'
                    f'<span style="color:#94a3b8;font-weight:600;">{row["pct"]}%</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No expense data for selected filters.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bar:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>📈 Monthly Income vs Expenses</div>", unsafe_allow_html=True)

        f_monthly = get_monthly_income_expense(filtered)
        if not f_monthly.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Income", x=f_monthly["month_str"], y=f_monthly["income"],
                marker_color="#06b6d4", opacity=0.85,
                hovertemplate="₹%{y:,.0f}<extra>Income</extra>",
            ))
            fig_bar.add_trace(go.Bar(
                name="Expenses", x=f_monthly["month_str"], y=f_monthly["expense"],
                marker_color="#8b5cf6", opacity=0.85,
                hovertemplate="₹%{y:,.0f}<extra>Expenses</extra>",
            ))
            fig_bar.update_layout(
                **CHART_LAYOUT,
                height=300,
                barmode="group",
                xaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)",
                           tickprefix="₹", tickformat=",.0f"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No monthly data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Expense Trend + Key Insights + Savings ─────────────────
    col_trend, col_insights = st.columns([6, 4])

    with col_trend:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>📉 Monthly Spending Trends</div>", unsafe_allow_html=True)

        f_trends = get_spending_trends(filtered)
        if not f_trends.empty:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                name="Expense", x=f_trends["month_str"], y=f_trends["expense"],
                mode="lines+markers",
                line=dict(color="#06b6d4", width=2.5),
                marker=dict(size=5, color="#06b6d4"),
                hovertemplate="₹%{y:,.0f}<extra>Expense</extra>",
            ))
            fig_line.add_trace(go.Scatter(
                name="Savings", x=f_trends["month_str"], y=f_trends["savings"],
                mode="lines+markers",
                line=dict(color="#10b981", width=2, dash="dot"),
                marker=dict(size=5, color="#10b981"),
                hovertemplate="₹%{y:,.0f}<extra>Savings</extra>",
            ))
            fig_line.add_trace(go.Scatter(
                name="3M Rolling Avg", x=f_trends["month_str"], y=f_trends["rolling_expense"],
                mode="lines",
                line=dict(color="#f97316", width=1.5, dash="dash"),
                hovertemplate="₹%{y:,.0f}<extra>3M Avg</extra>",
            ))
            fig_line.update_layout(
                **CHART_LAYOUT,
                height=280,
                xaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)",
                           tickprefix="₹", tickformat=",.0f"),
            )
            st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_insights:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>💡 Key Insights</div>", unsafe_allow_html=True)

        type_map = {"success": "insight-success", "warning": "insight-warning",
                    "danger": "insight-danger", "info": "insight-info"}
        for ins in insights[:4]:
            css = type_map.get(ins["type"], "insight-info")
            st.markdown(
                f'<div class="insight-card {css}">{ins["icon"]} {ins["text"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: Additional Stats ────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    stats = [
        ("💳 Avg Monthly Expense", f"₹{fkpis['avg_monthly_expense']:,.0f}"),
        ("📆 Months Covered",       str(fkpis["months_covered"])),
        ("🧾 Total Transactions",   str(fkpis["transaction_count"])),
        ("📈 Avg Monthly Income",   f"₹{fkpis['avg_monthly_income']:,.0f}"),
        ("💰 Avg Monthly Savings",  f"₹{fkpis['avg_monthly_income']-fkpis['avg_monthly_expense']:,.0f}"),
    ]
    for col, (label, val) in zip([s1, s2, s3, s4, s5], stats):
        with col:
            st.markdown(f"""
            <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                        border-radius:12px;padding:14px;text-align:center;">
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
                <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;margin-top:4px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)
