"""
models/train_model.py
Standalone script to train the expense categorizer and predictor.
Run: python models/train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_connection import initialize_database
from database.db_operations import get_all_transactions
from modules.expense_categorization import train_ml_model
from modules.expense_prediction import run_prediction


def main():
    print("🚀 FinSight AI – Model Training\n" + "=" * 40)
    initialize_database()

    df = get_all_transactions()
    if df.empty:
        print("❌ No transaction data found. Load data first.")
        return

    print(f"📊 Loaded {len(df)} transactions")

    # Train categorizer
    print("\n📐 Training Expense Categorizer...")
    result = train_ml_model(df)
    print(f"   Status: {result['status']}")
    print(f"   {result['message']}")

    # Train predictor
    print("\n🔮 Running Expense Predictor...")
    pred = run_prediction(df)
    print(f"   Status: {pred['status']}")
    if pred["status"] == "success":
        print(f"   Next month ({pred['next_month']}): ₹{pred['next_month_prediction']:,.0f}")
        metrics = pred["metrics"]
        if metrics:
            print(f"   MAE: ₹{metrics.get('mae', 0):,.0f} | RMSE: ₹{metrics.get('rmse', 0):,.0f}")
    else:
        print(f"   {pred.get('message', 'Error')}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
