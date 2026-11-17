"""
app.py
FinSight AI – Main Streamlit entry point.
Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Load .env if present ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": None,
        "About": "# FinSight AI\nAI-Powered Financial Expense Analyzer",
    },
)

# ── Inject CSS ────────────────────────────────────────────────────
css_path = ROOT / "assets" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── DB Init ───────────────────────────────────────────────────────
from database.db_connection import initialize_database
initialize_database()

# ── Load data ─────────────────────────────────────────────────────
from database.db_operations import get_all_transactions
import pandas as pd


@st.cache_data(ttl=30, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load all transactions from DB; cached for 30s."""
    return get_all_transactions()


# Invalidate cache when data is refreshed
if st.session_state.get("data_refreshed"):
    st.cache_data.clear()
    st.session_state["data_refreshed"] = False

df = load_data()

# ── Sidebar Navigation ─────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:20px 0 30px 0;text-align:left;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:12px;
                        width:44px;height:44px;display:flex;align-items:center;justify-content:center;
                        font-size:1.4rem;">💹</div>
            <div>
                <div style="color:#f1f5f9;font-size:1.2rem;font-weight:800;letter-spacing:-0.02em;">FinSight AI</div>
                <div style="color:#64748b;font-size:0.72rem;">AI Financial Assistant</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    NAV_ITEMS = {
        "🏠  Dashboard":      "Dashboard",
        "📤  Upload Data":    "Upload Data",
        "📈  Analytics":      "Analytics",
        "🤖  AI Insights":    "AI Insights",
        "🔮  Forecasting":    "Forecasting",
        "💬  AI Chatbot":     "AI Chatbot",
        "⚙️  Settings":       "Settings",
    }

    selected = st.radio(
        "Navigation",
        list(NAV_ITEMS.keys()),
        label_visibility="collapsed",
    )
    page = NAV_ITEMS[selected]

    # Sidebar footer stats
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid rgba(59,130,246,0.15);padding-top:16px;'>", unsafe_allow_html=True)

    txn_count = len(df)
    if txn_count > 0:
        date_range = f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}"
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#64748b;padding:0 4px;">
            📊 <b style="color:#94a3b8;">{txn_count}</b> transactions<br>
            📅 <span style="color:#94a3b8;">{date_range}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.75rem;color:#64748b;padding:0 4px;">
            📭 No data loaded yet.<br>
            Go to <b>Settings</b> → Load Demo.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Page Router ───────────────────────────────────────────────────
if page == "Dashboard":
    from pages.dashboard import show
    show(df)

elif page == "Upload Data":
    from pages.transactions import show
    show(df)

elif page == "Analytics":
    from pages.expense_analysis import show
    show(df)

elif page == "AI Insights":
    from pages.categorization import show
    show(df)

elif page == "Forecasting":
    from pages.prediction import show
    show(df)

elif page == "AI Chatbot":
    from pages.assistant import show
    show(df)

elif page == "Settings":
    from pages.settings import show
    show(df)
