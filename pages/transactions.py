"""
pages/transactions.py
MODULE 2 – Transaction management: upload, preview, search, edit, delete, add.
"""

import streamlit as st
import pandas as pd
from datetime import date, datetime

from modules.data_processing import parse_csv, validate_dataframe
from database.db_operations import (
    insert_transactions, get_all_transactions, get_transactions_by_filter,
    update_transaction_category, delete_transaction, add_transaction,
    get_category_names,
)


def show(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🧾 Transactions</h1>
        <p class="page-subtitle">Upload, browse, search, and manage your financial transactions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_browse, tab_add = st.tabs(["📤 Upload CSV", "📋 Browse & Edit", "➕ Add Transaction"])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 – Upload CSV
    # ═══════════════════════════════════════════════════════════════════
    with tab_upload:
        st.markdown("""
        <div style="background:rgba(15,35,64,0.5);border:1px solid rgba(59,130,246,0.2);
                    border-radius:14px;padding:20px;margin-bottom:20px;">
            <h4 style="color:#f1f5f9;margin:0 0 8px 0;">📋 Supported CSV Formats</h4>
            <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
                Your CSV should contain columns for <b>date</b>, <b>description</b>, and <b>amount</b>.
                Optional: <b>type</b> (income/expense), <b>category</b>, <b>payment_method</b>.
                Column names are auto-detected — common bank export formats are supported.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="Max file size: 50 MB. Supported: UTF-8 encoded CSV files.",
        )

        if uploaded:
            with st.spinner("⏳ Parsing and cleaning your data..."):
                try:
                    parsed_df, report = parse_csv(uploaded)
                    warnings = validate_dataframe(parsed_df)

                    st.success(f"✅ Parsed successfully: **{report['final_rows']}** valid transactions.")

                    if report["dropped_null"] > 0:
                        st.info(f"ℹ️ Dropped {report['dropped_null']} fully empty rows.")
                    if report["dropped_duplicate"] > 0:
                        st.warning(f"⚠️ Removed {report['dropped_duplicate']} duplicate transactions.")
                    if report["dropped_invalid_amount"] > 0:
                        st.warning(f"⚠️ Skipped {report['dropped_invalid_amount']} rows with invalid amounts.")
                    for w in warnings:
                        st.warning(w)

                    st.markdown("#### 👁️ Preview (first 10 rows)")
                    preview_cols = ["date", "description", "amount", "type", "category", "payment_method"]
                    available = [c for c in preview_cols if c in parsed_df.columns]
                    st.dataframe(
                        parsed_df[available].head(10).style.format({"amount": "₹{:,.2f}"}),
                        use_container_width=True,
                    )

                    if st.button("💾 Confirm & Save to Database", type="primary"):
                        with st.spinner("Saving..."):
                            count = insert_transactions(parsed_df, source="uploaded")
                        st.success(f"✅ Saved {count} transactions to database!")
                        st.session_state["data_refreshed"] = True
                        st.rerun()

                except ValueError as e:
                    st.error(f"❌ Failed to process CSV: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 – Browse & Edit
    # ═══════════════════════════════════════════════════════════════════
    with tab_browse:
        if df.empty:
            st.markdown("""
            <div style="text-align:center;padding:40px;">
                <div style="font-size:3rem;">📭</div>
                <p style="color:#64748b;">No transactions yet. Upload a CSV or add transactions manually.</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Filters ──────────────────────────────────────────────────
        st.markdown("#### 🔍 Search & Filter")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            search = st.text_input("🔎 Search description", placeholder="e.g. Swiggy, Salary...")
        with fc2:
            cats = ["All"] + sorted(df["category"].unique().tolist())
            sel_cat = st.selectbox("Category", cats)
        with fc3:
            types = ["All", "Income", "Expense"]
            sel_type = st.selectbox("Type", types)
        with fc4:
            min_d = df["date"].min().date()
            max_d = df["date"].max().date()
            date_rng = st.date_input("Date range", value=(min_d, max_d),
                                     min_value=min_d, max_value=max_d,
                                     label_visibility="visible")

        # Apply filters
        filtered = df.copy()
        if search:
            filtered = filtered[filtered["description"].str.contains(search, case=False, na=False)]
        if sel_cat != "All":
            filtered = filtered[filtered["category"] == sel_cat]
        if sel_type != "All":
            filtered = filtered[filtered["type"] == sel_type.lower()]
        if isinstance(date_rng, tuple) and len(date_rng) == 2:
            filtered = filtered[
                (filtered["date"].dt.date >= date_rng[0]) &
                (filtered["date"].dt.date <= date_rng[1])
            ]

        # Stats row
        exp_total = filtered[filtered["type"] == "expense"]["amount"].sum()
        inc_total = filtered[filtered["type"] == "income"]["amount"].sum()
        s1, s2, s3 = st.columns(3)
        s1.metric("🧾 Transactions", len(filtered))
        s2.metric("💸 Expenses",    f"₹{exp_total:,.0f}")
        s3.metric("💰 Income",      f"₹{inc_total:,.0f}")

        st.markdown("---")

        # ── Transaction Table ─────────────────────────────────────────
        if filtered.empty:
            st.info("No transactions match your filters.")
        else:
            category_list = get_category_names()
            display_cols = ["date", "description", "amount", "type", "category", "payment_method"]
            available    = [c for c in display_cols if c in filtered.columns]

            # Format for display
            disp_df = filtered[available].copy()
            disp_df["date"]   = pd.to_datetime(disp_df["date"]).dt.strftime("%d %b %Y")
            disp_df["amount"] = disp_df["amount"].apply(lambda x: f"₹{x:,.2f}")
            disp_df["type"]   = disp_df["type"].str.capitalize()

            st.dataframe(disp_df, use_container_width=True, height=400)

            # ── Edit / Delete panel ───────────────────────────────────
            st.markdown("#### ✏️ Edit or Delete a Transaction")
            with st.expander("Show edit controls"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    txn_ids = filtered["id"].tolist() if "id" in filtered.columns else []
                    if txn_ids:
                        sel_id = st.selectbox(
                            "Select Transaction ID (db id)",
                            txn_ids,
                            format_func=lambda x: f"ID {x} – " + filtered[filtered["id"] == x]["description"].values[0][:40],
                        )
                        new_cat = st.selectbox("New Category", category_list)
                        if st.button("💾 Update Category"):
                            update_transaction_category(sel_id, new_cat)
                            st.success(f"✅ Category updated to **{new_cat}**.")
                            st.session_state["data_refreshed"] = True
                            st.rerun()
                with ec2:
                    if txn_ids:
                        del_id = st.selectbox(
                            "Select to Delete",
                            txn_ids,
                            format_func=lambda x: f"ID {x} – " + filtered[filtered["id"] == x]["description"].values[0][:40],
                            key="del_select",
                        )
                        if st.button("🗑️ Delete Transaction", type="secondary"):
                            delete_transaction(del_id)
                            st.success("Transaction deleted.")
                            st.session_state["data_refreshed"] = True
                            st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3 – Add Transaction
    # ═══════════════════════════════════════════════════════════════════
    with tab_add:
        st.markdown("#### ➕ Add a New Transaction")
        with st.form("add_txn_form"):
            ac1, ac2 = st.columns(2)
            with ac1:
                new_date  = st.date_input("Date", value=date.today())
                new_desc  = st.text_input("Description", placeholder="e.g. Uber ride to airport")
                new_amt   = st.number_input("Amount (₹)", min_value=0.01, step=1.0, format="%.2f")
            with ac2:
                new_type   = st.selectbox("Type", ["Expense", "Income"])
                new_cat    = st.selectbox("Category", get_category_names())
                new_method = st.selectbox("Payment Method", [
                    "UPI", "Credit Card", "Debit Card", "Cash", "Net Banking",
                    "NEFT", "IMPS", "FASTag", "Auto-debit", "Other",
                ])

            submitted = st.form_submit_button("💾 Add Transaction", type="primary")
            if submitted:
                if not new_desc.strip():
                    st.error("Description cannot be empty.")
                else:
                    add_transaction(
                        date=str(new_date),
                        description=new_desc.strip(),
                        amount=float(new_amt),
                        txn_type=new_type,
                        category=new_cat,
                        payment_method=new_method,
                    )
                    st.success(f"✅ Transaction added: **{new_desc}** ₹{new_amt:,.2f}")
                    st.session_state["data_refreshed"] = True
