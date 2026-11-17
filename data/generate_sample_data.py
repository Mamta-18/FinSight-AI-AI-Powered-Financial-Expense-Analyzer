"""
data/generate_sample_data.py
Generates a realistic synthetic Indian personal finance dataset (2025).
Run this script once to create sample_transactions.csv.
"""

import csv
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
OUTPUT_PATH = Path(__file__).parent / "sample_transactions.csv"

CATEGORIES = {
    "Food & Dining": {
        "descs": [
            "Swiggy order - biryani", "Zomato pizza delivery", "Cafe Coffee Day",
            "McDonald's drive-through", "KFC bucket meal", "Domino's online order",
            "Hotel Saravana Bhavan", "Haldiram's snacks", "Starbucks coffee",
            "Local dhaba lunch", "Canteen dinner", "Udupi restaurant",
            "Barbeque Nation dinner", "Food court lunch", "Chai tapri",
            "Subway sandwich", "Chinese restaurant", "South Indian thali",
            "Momo stall", "Street food outing",
        ],
        "range": (120, 1200),
        "freq_per_month": 20,
        "payment": ["UPI", "Credit Card", "Debit Card", "Cash"],
    },
    "Groceries": {
        "descs": [
            "BigBasket order", "Blinkit grocery delivery", "DMart supermarket",
            "Reliance Fresh", "More Supermarket", "Zepto quick delivery",
            "Local kirana store", "JioMart vegetables", "Spencer's supermarket",
            "Fresh fruits purchase", "Weekly provisions", "Milk and dairy",
        ],
        "range": (300, 3500),
        "freq_per_month": 8,
        "payment": ["UPI", "Debit Card", "Credit Card", "Cash"],
    },
    "Shopping": {
        "descs": [
            "Amazon purchase", "Flipkart order", "Myntra clothing",
            "Ajio fashion", "Nykaa cosmetics", "Lifestyle store",
            "Westside apparel", "Shoppers Stop", "H&M purchase",
            "Zara outfit", "Electronics - Croma", "Books purchase",
            "Sports gear", "Home decor item", "Meesho order",
        ],
        "range": (200, 8000),
        "freq_per_month": 5,
        "payment": ["Credit Card", "Debit Card", "UPI", "Net Banking"],
    },
    "Transportation": {
        "descs": [
            "Uber ride to office", "Ola cab - airport", "Metro card recharge",
            "Rapido bike ride", "Petrol - HP Petrol Pump", "Diesel refill",
            "IRCTC train ticket", "Bus pass renewal", "FASTag toll payment",
            "Parking fee - mall", "Auto rickshaw", "Local train ticket",
            "Namma Yatri ride", "BluSmart EV cab",
        ],
        "range": (30, 2500),
        "freq_per_month": 15,
        "payment": ["UPI", "FASTag", "Cash", "Debit Card"],
    },
    "Bills & Utilities": {
        "descs": [
            "Electricity bill - BESCOM", "Water bill payment",
            "Airtel broadband bill", "Jio postpaid bill", "Vodafone mobile recharge",
            "Society maintenance", "Gas cylinder booking", "DTH recharge - Tata Play",
            "BSNL landline bill", "ACT Fibernet bill", "LPG gas bill",
        ],
        "range": (200, 4000),
        "freq_per_month": 4,
        "payment": ["Net Banking", "UPI", "Auto-debit"],
    },
    "Healthcare": {
        "descs": [
            "Apollo Pharmacy medicines", "1mg online order", "Doctor consultation fee",
            "Diagnostic lab test", "PharmEasy order", "Dental checkup",
            "Eye checkup - optician", "Gym membership", "Yoga class fee",
            "Physiotherapy session", "Health insurance premium",
        ],
        "range": (100, 5000),
        "freq_per_month": 2,
        "payment": ["UPI", "Credit Card", "Cash"],
    },
    "Entertainment": {
        "descs": [
            "Netflix subscription", "Amazon Prime Video", "Disney+ Hotstar",
            "Spotify Premium", "PVR movie tickets", "INOX cinema",
            "BookMyShow event", "YouTube Premium", "Cricket match tickets",
            "Gaming - Steam purchase", "Pub night out", "Karaoke evening",
        ],
        "range": (100, 3000),
        "freq_per_month": 4,
        "payment": ["Credit Card", "UPI", "Debit Card"],
    },
    "Education": {
        "descs": [
            "Udemy course purchase", "Coursera subscription", "College fee installment",
            "Byju's subscription", "Unacademy pro", "Coaching class fee",
            "Books & stationery", "Exam registration fee", "Workshop registration",
            "Skill certification", "Library membership",
        ],
        "range": (200, 15000),
        "freq_per_month": 1,
        "payment": ["Net Banking", "Credit Card", "UPI"],
    },
    "Travel": {
        "descs": [
            "IndiGo flight ticket", "Air India booking",
            "OYO hotel booking", "MakeMyTrip hotel",
            "Goibibo holiday package", "Airbnb stay",
            "Resort booking - Goa", "Travel insurance", "IRCTC Rajdhani ticket",
            "Vistara flight upgrade",
        ],
        "range": (800, 25000),
        "freq_per_month": 0.3,
        "payment": ["Credit Card", "Net Banking", "Debit Card"],
    },
}

INCOME_TEMPLATES = [
    {"desc": "Monthly Salary - Company", "range": (48000, 52000)},
    {"desc": "Freelance project payment", "range": (5000, 15000)},
    {"desc": "Interest credited - SBI Savings", "range": (200, 800)},
    {"desc": "Cashback credited", "range": (50, 500)},
    {"desc": "Dividend received - Mutual Fund", "range": (500, 2000)},
    {"desc": "Reimbursement - office expense", "range": (1000, 4000)},
]


def random_date_in_month(year: int, month: int) -> date:
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    start = date(year, month, 1)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


rows = []
txn_num = 1

for month in range(1, 13):  # Jan–Dec 2025
    year = 2025

    # ── Income ──────────────────────────────────────────────────────
    # Salary (always 1st of month)
    sal_range = INCOME_TEMPLATES[0]["range"]
    rows.append({
        "transaction_id": f"TXN{txn_num:04d}",
        "date": date(year, month, 1).isoformat(),
        "description": INCOME_TEMPLATES[0]["desc"],
        "amount": round(random.uniform(*sal_range), 2),
        "type": "income",
        "category": "Income",
        "payment_method": "NEFT",
    })
    txn_num += 1

    # Occasional extra income
    if random.random() < 0.4:
        tmpl = random.choice(INCOME_TEMPLATES[1:])
        rows.append({
            "transaction_id": f"TXN{txn_num:04d}",
            "date": random_date_in_month(year, month).isoformat(),
            "description": tmpl["desc"],
            "amount": round(random.uniform(*tmpl["range"]), 2),
            "type": "income",
            "category": "Income",
            "payment_method": random.choice(["NEFT", "UPI", "IMPS"]),
        })
        txn_num += 1

    # ── Expenses ─────────────────────────────────────────────────────
    for category, cfg in CATEGORIES.items():
        # Determine number of transactions this month
        base_freq = cfg["freq_per_month"]
        # Add slight month variation
        if isinstance(base_freq, float):
            count = 1 if random.random() < base_freq else 0
        else:
            count = max(0, int(base_freq + random.randint(-2, 3)))

        for _ in range(count):
            amount = round(random.uniform(*cfg["range"]), 2)
            # Festive months (Oct, Nov) → higher shopping
            if category == "Shopping" and month in [10, 11]:
                amount = round(amount * random.uniform(1.3, 2.0), 2)
            # Summer (Apr–Jun) → higher travel
            if category == "Travel" and month in [4, 5, 6, 12]:
                amount = round(amount * random.uniform(1.0, 2.5), 2)

            rows.append({
                "transaction_id": f"TXN{txn_num:04d}",
                "date": random_date_in_month(year, month).isoformat(),
                "description": random.choice(cfg["descs"]),
                "amount": amount,
                "type": "expense",
                "category": category,
                "payment_method": random.choice(cfg["payment"]),
            })
            txn_num += 1

# Sort by date
rows.sort(key=lambda r: r["date"])

# Write CSV
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fieldnames = ["transaction_id", "date", "description", "amount", "type", "category", "payment_method"]
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} transactions -> {OUTPUT_PATH}")
