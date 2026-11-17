"""
database/db_operations.py
CRUD operations for all FinSight AI tables.
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional
import pandas as pd
from database.db_connection import get_connection


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def insert_transactions(df: pd.DataFrame, source: str = "uploaded") -> int:
    """
    Bulk-insert a cleaned DataFrame of transactions.
    Returns number of rows inserted.
    """
    conn = get_connection()
    inserted = 0
    for _, row in df.iterrows():
        try:
            txn_id = row.get("transaction_id") or str(uuid.uuid4())[:8].upper()
            conn.execute(
                """
                INSERT OR IGNORE INTO transactions
                    (transaction_id, date, description, amount, type, category, payment_method, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    txn_id,
                    str(row.get("date", "")),
                    str(row.get("description", ""))[:300],
                    float(row.get("amount", 0)),
                    str(row.get("type", "expense")).lower(),
                    str(row.get("category", "Other")),
                    str(row.get("payment_method", "Unknown")),
                    source,
                ),
            )
            inserted += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    return inserted


def get_all_transactions() -> pd.DataFrame:
    """Return all transactions as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY date DESC",
        conn,
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df


def get_transactions_by_filter(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    txn_type: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:
    """Return filtered transactions."""
    query = "SELECT * FROM transactions WHERE 1=1"
    params: list = []

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    if txn_type and txn_type != "All":
        query += " AND type = ?"
        params.append(txn_type.lower())
    if search:
        query += " AND description LIKE ?"
        params.append(f"%{search}%")

    query += " ORDER BY date DESC"

    conn = get_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df


def update_transaction_category(txn_id: int, new_category: str) -> None:
    """Update the category of a transaction by its DB id."""
    conn = get_connection()
    conn.execute(
        "UPDATE transactions SET category = ? WHERE id = ?",
        (new_category, txn_id),
    )
    conn.commit()
    conn.close()


def delete_transaction(txn_id: int) -> None:
    """Delete a transaction by its DB id."""
    conn = get_connection()
    conn.execute("DELETE FROM transactions WHERE id = ?", (txn_id,))
    conn.commit()
    conn.close()


def add_transaction(
    date: str,
    description: str,
    amount: float,
    txn_type: str,
    category: str,
    payment_method: str,
) -> None:
    """Add a single transaction manually."""
    txn_id = str(uuid.uuid4())[:8].upper()
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO transactions
            (transaction_id, date, description, amount, type, category, payment_method, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'manual')
        """,
        (txn_id, date, description, amount, txn_type.lower(), category, payment_method),
    )
    conn.commit()
    conn.close()


def clear_all_transactions() -> None:
    """Delete all transactions (used in settings)."""
    conn = get_connection()
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


def get_transaction_count() -> int:
    """Return total number of transactions."""
    conn = get_connection()
    cur = conn.execute("SELECT COUNT(*) FROM transactions")
    count = cur.fetchone()[0]
    conn.close()
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_categories() -> list[dict]:
    """Return list of category dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM categories ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_category_names() -> list[str]:
    """Return just category names."""
    return [c["name"] for c in get_all_categories()]


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

def get_setting(key: str, default: str = "") -> str:
    """Get a user setting value."""
    conn = get_connection()
    row = conn.execute(
        "SELECT value FROM user_settings WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Upsert a user setting."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO user_settings (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_prediction(month: int, year: int, predicted: float, actual: Optional[float], version: str) -> None:
    """Upsert a model prediction."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO model_predictions (month, year, predicted_amount, actual_amount, model_version)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(month, year) DO UPDATE SET
            predicted_amount = excluded.predicted_amount,
            actual_amount    = excluded.actual_amount,
            model_version    = excluded.model_version,
            created_at       = datetime('now')
        """,
        (month, year, predicted, actual, version),
    )
    conn.commit()
    conn.close()


def get_predictions() -> pd.DataFrame:
    """Return all predictions."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM model_predictions ORDER BY year, month",
        conn,
    )
    conn.close()
    return df
