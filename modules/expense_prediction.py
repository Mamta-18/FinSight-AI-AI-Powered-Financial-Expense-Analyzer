"""
modules/expense_prediction.py
Time-series expense forecasting using Ridge Regression with lag/rolling features.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREDICTOR_PATH = Path(__file__).parent.parent / "models" / "saved_models" / "predictor.pkl"
MIN_MONTHS = 4   # Minimum months needed to train predictor


def _build_monthly_expense_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into monthly expense totals."""
    if df.empty or "date" not in df.columns or "type" not in df.columns:
        return pd.DataFrame(columns=["month", "expense", "month_str"])

    exp = df[df["type"] == "expense"].copy()
    if exp.empty:
        return pd.DataFrame(columns=["month", "expense", "month_str"])

    exp["date"] = pd.to_datetime(exp["date"], errors="coerce")
    exp = exp.dropna(subset=["date"]).copy()
    if exp.empty:
        return pd.DataFrame(columns=["month", "expense", "month_str"])

    exp["month"] = exp["date"].dt.to_period("M")
    monthly = exp.groupby("month")["amount"].sum().reset_index(name="expense")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["month_str"] = monthly["month"].astype(str)
    return monthly


def _chronological_validation(monthly: pd.DataFrame, model_name: str, predictions: pd.Series) -> dict:
    """Compute validation metrics on the last 20% of a chronological period."""
    if len(monthly) < 4:
        return {"mae": None, "rmse": None, "r2": None}

    split_idx = max(1, int(len(monthly) * 0.8))
    y_true = monthly["expense"].iloc[split_idx:].reset_index(drop=True)
    y_pred = predictions.iloc[split_idx:].reset_index(drop=True)

    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    return {
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
    }


def _build_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features for regression."""
    df = monthly.copy()
    df["month_num"]   = df["month"].apply(lambda p: p.month)
    df["year"]        = df["month"].apply(lambda p: p.year)
    df["month_index"] = range(len(df))

    # Lag features
    df["lag_1"] = df["expense"].shift(1)
    df["lag_2"] = df["expense"].shift(2)
    df["lag_3"] = df["expense"].shift(3)

    # Rolling statistics
    df["rolling_mean_3"] = df["expense"].shift(1).rolling(3, min_periods=1).mean()
    df["rolling_std_3"]  = df["expense"].shift(1).rolling(3, min_periods=1).std().fillna(0)

    # Seasonal features
    df["sin_month"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month_num"] / 12)

    return df


FEATURE_COLS = [
    "month_index", "month_num", "sin_month", "cos_month",
    "lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3",
]


def run_prediction(df: pd.DataFrame) -> dict:
    """
    Train predictor and return forecast results.
    Returns a dict with: status, monthly_data, next_month_prediction, metrics
    """
    if df.empty:
        return {"status": "no_data", "message": "No transaction data available."}

    monthly = _build_monthly_expense_series(df)

    if len(monthly) < MIN_MONTHS:
        return {
            "status": "insufficient_data",
            "message": (
                f"At least {MIN_MONTHS} months of expense history are needed to generate predictions. "
                f"Currently available: {len(monthly)} month(s)."
            ),
        }

    feat_df = _build_features(monthly)
    feat_df = feat_df.dropna(subset=FEATURE_COLS + ["expense"]).reset_index(drop=True)

    if len(feat_df) < 3:
        return {"status": "insufficient_data", "message": "Not enough complete feature rows after lag creation."}

    X = feat_df[FEATURE_COLS].values
    y = feat_df["expense"].values

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    candidates = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=None),
        "moving_average": None,
    }

    model_results = []

    for name, model in candidates.items():
        if name == "moving_average":
            if len(y_train) >= 2:
                moving_avg = pd.Series(y_train).rolling(window=min(3, len(y_train)), min_periods=1).mean().iloc[-1]
                y_test_series = pd.Series(y_test)
                pred_test = y_test_series.rolling(window=min(3, len(y_test_series)), min_periods=1).mean().fillna(y_test_series.iloc[0]).to_numpy()
                metrics = _chronological_validation(pd.DataFrame({"expense": y}), name, pd.Series(np.concatenate([np.array(y_train), pred_test])))
            else:
                continue
        else:
            model.fit(X_train, y_train)
            pred_test = model.predict(X_test)
            metrics = _chronological_validation(pd.DataFrame({"expense": y}), name, pd.Series(np.concatenate([np.array(y_train), pred_test])))

        if metrics["mae"] is not None:
            model_results.append(metrics)

    best_model = min(model_results, key=lambda item: item["mae"]) if model_results else None
    if best_model is None:
        return {"status": "insufficient_data", "message": "No valid model benchmark could be computed."}

    chosen_name = best_model["model"]
    if chosen_name == "ridge":
        final_model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    elif chosen_name == "random_forest":
        final_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=None)
    else:
        final_model = None

    if final_model is not None:
        final_model.fit(X_train, y_train)
        all_preds = final_model.predict(X)
        next_features = np.array([[
            feat_df["month_index"].iloc[-1] + 1,
            (monthly["month"].iloc[-1] + 1).month,
            np.sin(2 * np.pi * ((monthly["month"].iloc[-1] + 1).month) / 12),
            np.cos(2 * np.pi * ((monthly["month"].iloc[-1] + 1).month) / 12),
            y[-1],
            y[-2] if len(y) >= 2 else y[-1],
            y[-3] if len(y) >= 3 else y[-1],
            float(np.mean(y[-3:])),
            float(np.std(y[-3:])),
        ]])
        next_pred = float(final_model.predict(next_features)[0])
    else:
        all_preds = np.array([y[0]] * len(y))
        moving_avg = pd.Series(y).rolling(window=min(3, len(y)), min_periods=1).mean().iloc[-1]
        next_pred = float(max(0, moving_avg))

    next_pred = max(0, next_pred)

    PREDICTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    if final_model is not None:
        joblib.dump(final_model, PREDICTOR_PATH)

    result_df = feat_df[["month_str", "expense"]].copy()
    result_df["predicted"] = np.clip(all_preds, 0, None)
    result_df["split_point"] = ["train"] * split + ["test"] * (len(feat_df) - split)

    next_month_period = monthly["month"].iloc[-1] + 1
    next_row = pd.DataFrame([{
        "month_str": str(next_month_period),
        "expense":   None,
        "predicted": next_pred,
        "split_point": "forecast",
    }])
    result_df = pd.concat([result_df, next_row], ignore_index=True)

    last_actual = float(monthly["expense"].iloc[-1])
    pct_change = ((next_pred - last_actual) / last_actual * 100) if last_actual > 0 else 0

    return {
        "status": "success",
        "monthly_data": result_df,
        "next_month": str(next_month_period),
        "next_month_prediction": round(next_pred, 2),
        "last_actual": round(last_actual, 2),
        "pct_change": round(pct_change, 1),
        "metrics": {
            "mae": best_model["mae"],
            "rmse": best_model["rmse"],
            "r2": best_model["r2"],
        },
        "model_type": chosen_name.replace("_", " ").title(),
        "train_months": split,
        "test_months": len(X_test),
        "model_comparison": model_results,
    }
