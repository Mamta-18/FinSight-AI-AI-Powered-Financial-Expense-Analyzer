"""
pages/prediction.py
MODULE 5 – Expense Prediction using ML regression with time-series features.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.expense_prediction import run_prediction

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def show(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🔮 Expense Prediction</h1>
        <p class="page-subtitle">Machine learning forecast of your upcoming monthly expenses.</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("📭 No transaction data available. Upload your CSV first.")
        return

    # Info box
    st.markdown("""
    <div class="insight-card insight-info">
        🧮 <b>How it works:</b> The predictor trains a <b>Ridge/GradientBoosting</b> model using
        lag features, rolling averages, and seasonal signals derived from your monthly expense history.
        It uses a <b>chronological train/test split</b> to avoid data leakage.
        Requires at least <b>4 months</b> of history.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("⏳ Running prediction model..."):
        result = run_prediction(df)

    if result["status"] == "insufficient_data":
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;background:rgba(15,35,64,0.5);
                    border-radius:16px;border:1px dashed rgba(59,130,246,0.3);">
            <div style="font-size:4rem;">📊</div>
            <h3 style="color:#f1f5f9;">Not Enough Data</h3>
            <p style="color:#64748b;">{result['message']}</p>
        </div>
        """, unsafe_allow_html=True)
        return

    if result["status"] == "no_data":
        st.error(result["message"])
        return

    # ── Forecast Card ──────────────────────────────────────────────
    next_pred = result["next_month_prediction"]
    last_act  = result["last_actual"]
    pct       = result["pct_change"]
    pct_color = "#f43f5e" if pct > 0 else "#10b981"
    pct_arrow = "↑" if pct > 0 else "↓"

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(f"""
        <div class="prediction-card">
            <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.08em;">
                Predicted Next Month
            </div>
            <div style="color:#f1f5f9;font-size:2.4rem;font-weight:800;margin:8px 0;">₹{next_pred:,.0f}</div>
            <div style="color:{pct_color};font-weight:700;">{pct_arrow} {abs(pct):.1f}% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.metric("Next Month", result["next_month"])
    with fc3:
        st.metric("Last Actual Expense", f"₹{last_act:,.0f}")
    with fc4:
        st.metric("Model Type", result["model_type"])

    # ── Model Metrics ──────────────────────────────────────────────
    metrics = result["metrics"]
    if metrics:
        st.markdown("#### 📐 Model Evaluation (Test Set)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE",           f"₹{metrics.get('mae', 0):,.0f}")
        m2.metric("RMSE",          f"₹{metrics.get('rmse', 0):,.0f}")
        m3.metric("R²",            f"{metrics.get('r2', 'N/A')}" if metrics.get("r2") is not None else "N/A")
        m4.metric("Train Months",  result["train_months"])

    # ── Prediction Chart ───────────────────────────────────────────
    st.markdown("#### 📈 Actual vs Predicted Expenses")
    monthly_data = result["monthly_data"]

    actual_df  = monthly_data[monthly_data["expense"].notna()]
    pred_df    = monthly_data.copy()
    forecast   = monthly_data[monthly_data["split_point"] == "forecast"]

    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        name="Actual Expense",
        x=actual_df["month_str"],
        y=actual_df["expense"],
        mode="lines+markers",
        line=dict(color="#06b6d4", width=2.5),
        marker=dict(size=7, color="#06b6d4"),
        hovertemplate="<b>%{x}</b><br>Actual: ₹%{y:,.0f}<extra></extra>",
    ))

    # Predicted line (all, includes train, test, forecast)
    fig.add_trace(go.Scatter(
        name="Predicted Expense",
        x=pred_df["month_str"],
        y=pred_df["predicted"],
        mode="lines+markers",
        line=dict(color="#a855f7", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond", color="#a855f7"),
        hovertemplate="<b>%{x}</b><br>Predicted: ₹%{y:,.0f}<extra></extra>",
    ))

    # Forecast marker
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            name="Forecast",
            x=forecast["month_str"],
            y=forecast["predicted"],
            mode="markers",
            marker=dict(size=14, color="#f97316", symbol="star"),
            hovertemplate="<b>%{x}</b><br>Forecast: ₹%{y:,.0f}<extra></extra>",
        ))

    # Shade forecast region
    if not forecast.empty:
        fig.add_vrect(
            x0=forecast["month_str"].iloc[0],
            x1=forecast["month_str"].iloc[-1],
            fillcolor="rgba(249,115,22,0.08)",
            line_width=0,
            annotation_text="Forecast",
            annotation_position="top left",
            annotation_font_color="#f97316",
        )

    fig.update_layout(
        **CHART_LAYOUT,
        height=380,
        title="Actual vs Predicted Monthly Expenses",
        title_font=dict(color="#f1f5f9", size=14),
        xaxis=dict(tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(tickprefix="₹", tickformat=",.0f", gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Data table ──────────────────────────────────────────────────
    st.markdown("#### 📋 Monthly Data Table")
    tbl = monthly_data.copy()
    tbl["expense"]   = tbl["expense"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    tbl["predicted"] = tbl["predicted"].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
    tbl.columns = [c.replace("_"," ").title() for c in tbl.columns]
    st.dataframe(tbl, use_container_width=True)

    # ── Disclaimer ──────────────────────────────────────────────────
    st.markdown("""
    <div style="color:#64748b;font-size:0.78rem;margin-top:12px;padding:10px;
                background:rgba(100,116,139,0.08);border-radius:8px;">
        ⚠️ Predictions are based on historical patterns and are for informational purposes only.
        Actual expenses may vary. This is not financial advice.
    </div>
    """, unsafe_allow_html=True)
