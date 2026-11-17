"""
modules/expense_categorization.py
Rule-based + ML (RandomForest + TF-IDF) expense categorizer for FinSight AI.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


MODELS_DIR = Path(__file__).parent.parent / "models" / "saved_models"
MODEL_PATH = MODELS_DIR / "categorizer.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
MIN_TRAINING_SAMPLES = 50

# ── Keyword Rules ──────────────────────────────────────────────────────────────
CATEGORY_RULES: dict[str, list[str]] = {
    "Food & Dining": [
        "restaurant", "cafe", "coffee", "pizza", "burger", "biryani", "swiggy",
        "zomato", "domino", "kfc", "mcdonald", "subway", "hotel food", "dine",
        "lunch", "dinner", "breakfast", "chai", "dhaba", "canteen", "food court",
        "barbeque", "sushi", "meal", "tiffin", "dabba", "udupi", "mess",
    ],
    "Groceries": [
        "grocery", "supermarket", "bigbasket", "blinkit", "zepto", "jiomart",
        "dmart", "reliance fresh", "more supermarket", "vegetables", "fruits",
        "milk", "bread", "eggs", "rice", "wheat", "pulses", "dal", "ration",
        "kirana", "general store", "provisions",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "mall", "shopping",
        "clothes", "shirt", "jeans", "shoes", "apparel", "fashion", "h&m",
        "zara", "lifestyle", "westside", "shoppers stop", "max fashion",
        "meesho", "snapdeal", "tata cliq", "croma", "vijay sales",
    ],
    "Transportation": [
        "uber", "ola", "rapido", "auto", "taxi", "metro", "bus", "train",
        "irctc", "petrol", "diesel", "fuel", "parking", "toll", "fastag",
        "rickshaw", "cab", "ride", "commute", "transport", "vehicle",
    ],
    "Bills & Utilities": [
        "electricity", "water", "gas", "internet", "broadband", "wifi", "mobile",
        "phone bill", "recharge", "airtel", "jio", "vodafone", "bsnl", "act",
        "postpaid", "utility", "bill payment", "maintenance", "society",
        "cable", "dth",
    ],
    "Healthcare": [
        "hospital", "clinic", "doctor", "medical", "pharmacy", "medicine",
        "apollo", "fortis", "max hospital", "chemist", "lab", "diagnostic",
        "health check", "dentist", "optician", "physiotherapy", "ayurveda",
        "netmeds", "1mg", "pharmeasy", "insurance premium health",
    ],
    "Entertainment": [
        "netflix", "prime video", "disney", "hotstar", "spotify", "youtube",
        "movie", "cinema", "pvr", "inox", "bookmyshow", "concert", "event",
        "game", "gaming", "pubg", "cricket match", "sports", "club",
        "bar", "pub", "lounge", "nightclub", "karaoke",
    ],
    "Education": [
        "school", "college", "university", "tuition", "coaching", "udemy",
        "coursera", "edtech", "byju", "unacademy", "vedantu", "book",
        "stationery", "library", "exam fee", "course", "workshop", "seminar",
        "skill", "certification",
    ],
    "Travel": [
        "flight", "air india", "indigo", "spicejet", "vistara", "hotel",
        "oyo", "makemytrip", "yatra", "goibibo", "booking.com", "airbnb",
        "resort", "tour", "trip", "holiday", "vacation", "visa", "passport",
        "luggage", "travel insurance",
    ],
    "Income": [
        "salary", "wages", "stipend", "bonus", "credit", "deposit",
        "refund", "cashback", "reimbursement", "dividend", "interest credited",
        "transfer received", "income",
    ],
}


def _preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize spaces."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_based_categorize(description: str) -> tuple[str, float]:
    """
    Categorize a transaction using keyword rules.
    Returns: (category, confidence)
    """
    text = _preprocess_text(description)
    tokens = set(text.split())

    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_RULES.items():
        matches = sum(1 for kw in keywords if kw in text or kw in tokens)
        if matches > 0:
            scores[category] = matches

    if not scores:
        return "Other", 0.4

    best = max(scores, key=scores.__getitem__)
    total = sum(scores.values())
    confidence = min(0.95, scores[best] / max(total, 1) + 0.5)
    return best, round(confidence, 2)


def categorize_dataframe(df: pd.DataFrame, use_ml: bool = True) -> pd.DataFrame:
    """
    Categorize all transactions in a DataFrame.
    Adds 'predicted_category', 'confidence', 'method' columns.
    """
    df = df.copy()
    df["predicted_category"] = "Other"
    df["confidence"] = 0.4
    df["method"] = "rule-based"

    # Try ML model first if available and requested
    ml_available = use_ml and MODEL_PATH.exists()
    ml_pipeline = None
    if ml_available:
        try:
            ml_pipeline = joblib.load(MODEL_PATH)
        except Exception:
            ml_available = False

    for idx, row in df.iterrows():
        desc = str(row.get("description", ""))

        if ml_available and ml_pipeline is not None:
            try:
                proba = ml_pipeline.predict_proba([_preprocess_text(desc)])[0]
                confidence = float(proba.max())
                if confidence >= 0.6:
                    label_enc = joblib.load(ENCODER_PATH)
                    pred_idx = int(proba.argmax())
                    category = label_enc.inverse_transform([pred_idx])[0]
                    df.at[idx, "predicted_category"] = category
                    df.at[idx, "confidence"] = round(confidence, 2)
                    df.at[idx, "method"] = "ML"
                    continue
            except Exception:
                pass

        # Fallback to rule-based
        cat, conf = rule_based_categorize(desc)
        df.at[idx, "predicted_category"] = cat
        df.at[idx, "confidence"] = conf
        df.at[idx, "method"] = "rule-based"

    return df


def train_ml_model(df: pd.DataFrame) -> dict:
    """
    Train a RandomForest + TF-IDF categorizer on labeled transaction data.
    Returns a report dict with both cross-validation and independent test metrics.
    """
    labeled = df[
        (df["type"] == "expense") &
        (df["category"].notna()) &
        (df["category"] != "Other")
    ].copy()

    if len(labeled) < MIN_TRAINING_SAMPLES:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_TRAINING_SAMPLES} labeled examples. Found {len(labeled)}.",
        }

    X = labeled["description"].apply(_preprocess_text).tolist()
    y = labeled["category"].tolist()

    if len(set(y)) < 2:
        return {
            "status": "insufficient_data",
            "message": "Need at least two distinct categories for model evaluation.",
        }

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_splits = min(5, min(np.bincount(y_enc).min(), 5))
    if n_splits < 2:
        return {
            "status": "insufficient_data",
            "message": "Not enough samples per category for a stratified evaluation split.",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf",   RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    cv_mean = float(cv_scores.mean())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    test_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(le.classes_)),
        target_names=list(le.classes_),
        output_dict=True,
        zero_division=0,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    return {
        "status": "success",
        "samples": len(labeled),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "categories": list(le.classes_),
        "cv_accuracy": round(cv_mean * 100, 1),
        "test_accuracy": round(test_accuracy, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_f1": round(test_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "message": (
            f"Model trained on {len(labeled)} samples. "
            f"CV accuracy: {round(cv_mean * 100, 1)}%. "
            f"Independent test F1: {round(test_f1, 4)}."
        ),
    }


def get_model_status() -> dict:
    """Return whether a trained ML model exists."""
    if MODEL_PATH.exists():
        stat = MODEL_PATH.stat()
        return {
            "available": True,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
        }
    return {"available": False}
