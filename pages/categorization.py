"""
pages/categorization.py
MODULE 3 – AI Expense Categorization: rule-based + ML model training UI.
"""

import streamlit as st
import pandas as pd
from modules.expense_categorization import (
    categorize_dataframe, train_ml_model, get_model_status,
    rule_based_categorize,
)
from database.db_operations import (
    get_all_transactions, update_transaction_category, get_category_names,
)


def show(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🤖 AI Categorization</h1>
        <p class="page-subtitle">Auto-categorize transactions using rule-based logic and ML classification.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏷️ Auto-Categorize", "🧠 Train ML Model", "🔬 Test Classifier"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 – Auto-Categorize transactions
    # ════════════════════════════════════════════════════════════════
    with tab1:
        model_status = get_model_status()

        # Model status banner
        if model_status["available"]:
            st.markdown(f"""
            <div class="insight-card insight-success">
                🧠 <b>ML Model Active</b> — Trained model available ({model_status['size_kb']} KB,
                modified {model_status['modified']}). High-confidence predictions will use ML.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card insight-info">
                📐 <b>Rule-Based Mode</b> — No trained ML model found. Using keyword-matching rules.
                Train the ML model in the <b>Train ML Model</b> tab for better accuracy.
            </div>
            """, unsafe_allow_html=True)

        if df.empty:
            st.info("📭 No transactions available. Upload data on the Transactions page.")
            return

        # Categorize sample
        expense_df = df[df["type"] == "expense"].copy()
        n_uncategorized = (expense_df["category"] == "Other").sum()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Expense Transactions", len(expense_df))
        col_b.metric("Uncategorized (Other)", n_uncategorized)
        col_c.metric("Model Type", "ML + Rules" if model_status["available"] else "Rules Only")

        st.markdown("---")
        st.markdown("#### 🏷️ Run Auto-Categorization")

        use_ml = model_status["available"]
        sample_size = st.slider("Transactions to preview", 10, min(200, len(expense_df)), 50)

        if st.button("▶️ Run Categorization (Preview)", type="primary"):
            with st.spinner("⏳ Categorizing transactions..."):
                sample = expense_df.head(sample_size).copy()
                result = categorize_dataframe(sample, use_ml=use_ml)

            # Display results
            st.markdown(f"#### Results for {len(result)} transactions")
            display_df = result[["date", "description", "amount", "category",
                                  "predicted_category", "confidence", "method"]].copy()
            display_df["date"]   = pd.to_datetime(display_df["date"]).dt.strftime("%d %b %Y")
            display_df["amount"] = display_df["amount"].apply(lambda x: f"₹{x:,.0f}")
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.0f}%")
            display_df["match"] = (
                display_df["category"] == result["predicted_category"]
            ).map({True: "✅", False: "🔄"})

            # Color confidence
            st.dataframe(display_df, use_container_width=True, height=400)

            # Stats
            ml_count   = (result["method"] == "ML").sum()
            rule_count = (result["method"] == "rule-based").sum()
            avg_conf   = result["confidence"].mean()
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("ML Categorized",    ml_count)
            sc2.metric("Rule Categorized",  rule_count)
            sc3.metric("Avg Confidence",    f"{avg_conf*100:.1f}%")

        st.markdown("---")
        st.markdown("#### 💾 Apply Categories to All Transactions")

        with st.expander("⚠️ This will update all 'Other' category transactions in the database"):
            confirm = st.checkbox("I understand this will modify the database")
            if confirm and st.button("🔄 Apply Categorization to All", type="primary"):
                uncategorized = expense_df[expense_df["category"] == "Other"].copy()
                if uncategorized.empty:
                    st.info("No uncategorized transactions found.")
                else:
                    with st.spinner(f"Categorizing {len(uncategorized)} transactions..."):
                        categorized = categorize_dataframe(uncategorized, use_ml=use_ml)
                        updated = 0
                        for _, row in categorized.iterrows():
                            if row["predicted_category"] != "Other" and "id" in row:
                                update_transaction_category(int(row["id"]), row["predicted_category"])
                                updated += 1
                    st.success(f"✅ Updated {updated} transactions.")
                    st.session_state["data_refreshed"] = True

    # ════════════════════════════════════════════════════════════════
    # TAB 2 – Train ML Model
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 🧠 Train RandomForest + TF-IDF Classifier")

        st.markdown("""
        <div class="insight-card insight-info">
            ℹ️ The ML model learns from your <b>labeled expense transactions</b>
            (transactions where category ≠ 'Other'). At least <b>50 labeled examples</b>
            are required. The model uses <b>TF-IDF</b> features on descriptions and
            <b>5-fold cross-validation</b> for evaluation.
        </div>
        """, unsafe_allow_html=True)

        if df.empty:
            st.info("No data available.")
            return

        labeled = df[(df["type"] == "expense") & (df["category"] != "Other")]
        st.markdown(f"""
        <div style="display:flex;gap:20px;margin:16px 0;">
            <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                        border-radius:12px;padding:16px 24px;text-align:center;">
                <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">Total Expenses</div>
                <div style="color:#f1f5f9;font-size:1.8rem;font-weight:800;">{len(df[df['type']=='expense'])}</div>
            </div>
            <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                        border-radius:12px;padding:16px 24px;text-align:center;">
                <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">Labeled Examples</div>
                <div style="color:#{'10b981' if len(labeled)>=50 else 'f59e0b'};font-size:1.8rem;font-weight:800;">{len(labeled)}</div>
            </div>
            <div style="background:rgba(15,35,64,0.6);border:1px solid rgba(59,130,246,0.2);
                        border-radius:12px;padding:16px 24px;text-align:center;">
                <div style="color:#64748b;font-size:0.75rem;text-transform:uppercase;">Unique Categories</div>
                <div style="color:#f1f5f9;font-size:1.8rem;font-weight:800;">{labeled['category'].nunique() if not labeled.empty else 0}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(labeled) < 50:
            st.warning(f"⚠️ Only {len(labeled)} labeled examples available. Need at least 50 to train.")
            st.info("💡 Tip: Run Auto-Categorize on the previous tab to label transactions first, then come back to train.")
        else:
            if st.button("🚀 Train ML Model", type="primary"):
                with st.spinner("⏳ Training model (this may take a moment)..."):
                    result = train_ml_model(df)

                if result["status"] == "success":
                    st.success(f"✅ {result['message']}")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Training Samples", result["samples"])
                    mc2.metric("CV Accuracy",      f"{result['cv_accuracy']}%")
                    mc3.metric("Categories",        len(result["categories"]))

                    st.markdown("**Learned Categories:**")
                    cats_html = " ".join(
                        f'<span style="background:rgba(59,130,246,0.2);color:#93c5fd;'
                        f'padding:3px 10px;border-radius:12px;font-size:0.8rem;margin:2px;">{c}</span>'
                        for c in result["categories"]
                    )
                    st.markdown(cats_html, unsafe_allow_html=True)
                else:
                    st.error(f"❌ {result['message']}")

        # Model info
        model_status = get_model_status()
        if model_status["available"]:
            st.markdown("---")
            st.markdown(f"""
            <div class="insight-card insight-success">
                ✅ Saved model: <b>{model_status['size_kb']} KB</b>, last trained {model_status['modified']}
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3 – Test Classifier
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 🔬 Test the Categorizer Live")
        st.markdown("Enter a transaction description to see how it gets categorized.")

        test_desc = st.text_input(
            "Transaction Description",
            placeholder="e.g. 'Uber ride to airport' or 'Monthly Netflix subscription'",
        )

        if test_desc:
            cat, conf = rule_based_categorize(test_desc)

            # Try ML too
            model_status = get_model_status()
            ml_result = None
            if model_status["available"]:
                try:
                    import joblib
                    from pathlib import Path
                    from modules.expense_categorization import _preprocess_text, ENCODER_PATH, MODEL_PATH
                    pipeline = joblib.load(MODEL_PATH)
                    le       = joblib.load(ENCODER_PATH)
                    proba    = pipeline.predict_proba([_preprocess_text(test_desc)])[0]
                    ml_conf  = float(proba.max())
                    ml_cat   = le.inverse_transform([int(proba.argmax())])[0]
                    ml_result = (ml_cat, ml_conf)
                except Exception:
                    pass

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"""
                <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
                            border-radius:12px;padding:20px;text-align:center;">
                    <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.06em;">Rule-Based Result</div>
                    <div style="color:#f1f5f9;font-size:1.6rem;font-weight:800;margin:8px 0;">{cat}</div>
                    <div style="color:#06b6d4;font-weight:600;">{conf*100:.0f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with tc2:
                if ml_result:
                    ml_cat, ml_conf = ml_result
                    st.markdown(f"""
                    <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);
                                border-radius:12px;padding:20px;text-align:center;">
                        <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.06em;">ML Model Result</div>
                        <div style="color:#f1f5f9;font-size:1.6rem;font-weight:800;margin:8px 0;">{ml_cat}</div>
                        <div style="color:#8b5cf6;font-weight:600;">{ml_conf*100:.0f}% confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:rgba(100,116,139,0.1);border:1px solid rgba(100,116,139,0.3);
                                border-radius:12px;padding:20px;text-align:center;">
                        <div style="color:#64748b;font-size:0.9rem;">ML model not trained yet.</div>
                        <div style="color:#94a3b8;font-size:0.8rem;margin-top:8px;">Train it in the previous tab.</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Keyword rules reference
        with st.expander("📋 View Keyword Rules"):
            from modules.expense_categorization import CATEGORY_RULES
            for cat_name, keywords in CATEGORY_RULES.items():
                st.markdown(f"**{cat_name}**: {', '.join(keywords[:10])}{'...' if len(keywords) > 10 else ''}")
