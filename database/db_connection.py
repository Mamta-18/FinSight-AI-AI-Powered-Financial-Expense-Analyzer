"""
database/db_connection.py
SQLite connection manager for FinSight AI.
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "database.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database() -> None:
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── Transactions ──────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id  TEXT    UNIQUE,
            date            TEXT    NOT NULL,
            description     TEXT    NOT NULL,
            amount          REAL    NOT NULL,
            type            TEXT    NOT NULL CHECK(type IN ('income','expense')),
            category        TEXT    DEFAULT 'Other',
            payment_method  TEXT    DEFAULT 'Unknown',
            source          TEXT    DEFAULT 'uploaded',
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Categories ────────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT    UNIQUE NOT NULL,
            color   TEXT    DEFAULT '#6366f1',
            icon    TEXT    DEFAULT '📂'
        )
    """)

    # ── User Settings ─────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            key     TEXT PRIMARY KEY,
            value   TEXT
        )
    """)

    # ── Model Predictions ─────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            month            INTEGER NOT NULL,
            year             INTEGER NOT NULL,
            predicted_amount REAL,
            actual_amount    REAL,
            model_version    TEXT,
            created_at       TEXT    DEFAULT (datetime('now')),
            UNIQUE(month, year)
        )
    """)

    # ── Seed default categories ───────────────────────────────────────────────
    default_categories = [
        ("Food & Dining",    "#f97316", "🍽️"),
        ("Shopping",         "#a855f7", "🛍️"),
        ("Transportation",   "#06b6d4", "🚗"),
        ("Bills & Utilities","#3b82f6", "⚡"),
        ("Healthcare",       "#10b981", "🏥"),
        ("Entertainment",    "#f43f5e", "🎬"),
        ("Education",        "#8b5cf6", "📚"),
        ("Travel",           "#0ea5e9", "✈️"),
        ("Groceries",        "#84cc16", "🛒"),
        ("Income",           "#22c55e", "💰"),
        ("Other",            "#6b7280", "📂"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO categories (name, color, icon) VALUES (?,?,?)",
        default_categories,
    )

    conn.commit()
    conn.close()
