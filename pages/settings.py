"""
pages/settings.py
Application settings: API keys, data management, theme preferences.
"""

import streamlit as st
import os
from pathlib import Path
from database.db_operations import (
    get_setting, set_setting, clear_all_transactions, get_transaction_count,
)
from database.db_connection import initialize_database


def show(df):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">⚙️ Settings</h1>
        <p class="page-subtitle">Configure AI providers, manage data, and customise the application.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🤖 AI Configuration", "🗄️ Data Management", "ℹ️ About"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 – AI Configuration
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        <div class="insight-card insight-info">
            🔒 <b>Security:</b> API keys are stored in the local SQLite database and
            never logged or transmitted beyond the configured LLM API endpoint.
            Only aggregated financial summaries are sent to the LLM — not raw transactions.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🤖 LLM Provider")
        current_provider = get_setting("llm_provider", "groq")
        provider = st.selectbox(
            "Select AI Provider",
            ["groq", "openai", "google"],
            index=["groq", "openai", "google"].index(current_provider) if current_provider in ["groq","openai","google"] else 0,
            help="Groq offers a free tier at console.groq.com. OpenAI and Google require paid API keys.",
        )

        st.markdown("---")
        st.markdown("#### 🔑 API Keys")
        st.caption(
            "Keys are stored locally in SQLite and are never shown again after saving. "
            "Enter a newly generated key to add or replace one. Leave a field blank to keep its saved key; "
            "a matching environment variable takes precedence over the saved value."
        )

        with st.form("api_keys_form"):
            groq_key = st.text_input(
                "Groq API Key", value="", type="password",
                help="Enter a newly generated key from console.groq.com to add or replace the saved key. "
                     "After an HTTP 401, replace it here and save.",
                placeholder="Enter new Groq key to replace saved key",
            )
            openai_key = st.text_input(
                "OpenAI API Key", value="", type="password",
                help="Enter a key from platform.openai.com to add or replace the saved key.",
                placeholder="Enter new OpenAI key to replace saved key",
            )
            google_key = st.text_input(
                "Google Gemini API Key", value="", type="password",
                help="Enter a key from aistudio.google.com to add or replace the saved key.",
                placeholder="Enter new Gemini key to replace saved key",
            )

            if st.form_submit_button("💾 Save API Keys", type="primary"):
                set_setting("llm_provider", provider)
                if groq_key:
                    set_setting("groq_api_key", groq_key)
                if openai_key:
                    set_setting("openai_api_key", openai_key)
                if google_key:
                    set_setting("google_api_key", google_key)
                st.success("✅ Settings saved!")

        # Test connection
        st.markdown("---")
        st.markdown("#### 🔌 Test Connection")
        if st.button("🧪 Test LLM Connection"):
            from modules.ai_assistant import test_llm_connection
            connected, message = test_llm_connection()
            if connected:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

        # Env var info
        with st.expander("📋 Environment Variable Setup (Alternative)"):
            st.markdown("""
            Create a `.env` file in the project root (copy from `.env.example`):
            ```env
            LLM_PROVIDER=groq
            GROQ_API_KEY=your_key_here
            # or
            OPENAI_API_KEY=your_key_here
            # or
            GOOGLE_API_KEY=your_key_here
            ```
            Environment variables take precedence over DB settings.
            """)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 – Data Management
    # ════════════════════════════════════════════════════════════════
    with tab2:
        txn_count = get_transaction_count()
        st.metric("Total Transactions in Database", txn_count)

        st.markdown("---")
        st.markdown("#### 📥 Load Demo Dataset")
        st.markdown("""
        <div class="demo-banner">
            📊 The demo dataset contains <b>500+ synthetic Indian personal finance transactions</b>
            for 2025 (Jan–Dec). It is clearly labeled as demo data and is meant for testing.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Load Demo Dataset", type="primary"):
            demo_path = Path(__file__).parent.parent / "data" / "sample_transactions.csv"
            if not demo_path.exists():
                # Generate it
                try:
                    import subprocess, sys
                    gen_script = Path(__file__).parent.parent / "data" / "generate_sample_data.py"
                    subprocess.run([sys.executable, str(gen_script)], check=True)
                except Exception as e:
                    st.error(f"Could not generate demo data: {e}")
                    return

            if demo_path.exists():
                from modules.data_processing import parse_csv
                from database.db_operations import insert_transactions
                try:
                    demo_df, report = parse_csv(str(demo_path))
                    count = insert_transactions(demo_df, source="demo")
                    st.success(f"✅ Loaded {count} demo transactions!")
                    st.session_state["data_refreshed"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading demo data: {e}")
            else:
                st.error("Demo data file not found. Run `python data/generate_sample_data.py` first.")

        st.markdown("---")
        st.markdown("#### 🗑️ Clear Data")
        with st.expander("⚠️ Danger Zone"):
            st.warning("This action is irreversible. All transaction data will be permanently deleted.")
            confirm = st.checkbox("I understand this will permanently delete all data")
            if confirm and st.button("🗑️ Delete All Transactions", type="secondary"):
                clear_all_transactions()
                st.success("✅ All transactions deleted.")
                st.session_state["data_refreshed"] = True
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🔧 Database Info")
        db_path = Path(__file__).parent.parent / "data" / "database.db"
        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            st.markdown(f"""
            <div style="background:rgba(15,35,64,0.5);border:1px solid rgba(59,130,246,0.2);
                        border-radius:10px;padding:14px;font-size:0.85rem;color:#94a3b8;">
                📁 Database: <code>{db_path}</code><br>
                📦 Size: {size_kb:.1f} KB<br>
                🧾 Transactions: {txn_count}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Database will be created on first use.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 – About
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <div style="font-size:4rem;margin-bottom:12px;">💹</div>
            <h2 style="color:#f1f5f9;font-size:2rem;font-weight:800;">FinSight AI</h2>
            <p style="color:#8b5cf6;font-weight:600;">Track Smarter. Spend Better. Predict Your Financial Future.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🛠️ Technology Stack**
            - Python 3.14
            - Streamlit
            - Pandas + NumPy
            - Scikit-learn (ML)
            - Plotly (Charts)
            - SQLite (Database)
            - OpenAI / Groq / Google Gemini (LLM)
            """)
        with col2:
            st.markdown("""
            **🤖 ML Models**
            - RandomForest + TF-IDF (Categorization)
            - Ridge / GradientBoosting (Expense Prediction)
            - Rule-based keyword classifier (Baseline)

            **📊 Features**
            - 8 interactive pages
            - Real-time analytics
            - AI-powered insights
            - Modular architecture
            """)

        st.markdown("""
        ---
        <div style="text-align:center;color:#64748b;font-size:0.8rem;">
            Built with ❤️ as a portfolio project demonstrating Python, SQL, ML, and GenAI skills.
            <br>⚠️ Not a licensed financial advisor. All insights are informational only.
        </div>
        """, unsafe_allow_html=True)
