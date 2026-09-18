"""
01_generate_data.py
Generates a realistic synthetic retail sales dataset covering 2 years,
5 stores, 6 product categories, with seasonality, promotions, and noise.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Configuration ----
start_date = "2024-01-01"
end_date = "2025-12-31"
dates = pd.date_range(start_date, end_date, freq="D")

stores = ["Store_A", "Store_B", "Store_C", "Store_D", "Store_E"]
store_base_traffic = {"Store_A": 220, "Store_B": 180, "Store_C": 260, "Store_D": 140, "Store_E": 200}

categories = {
    "Electronics": {"base_price": 120, "price_std": 30, "base_units": 8},
    "Groceries":   {"base_price": 15,  "price_std": 5,  "base_units": 40},
    "Clothing":    {"base_price": 35,  "price_std": 10, "base_units": 18},
    "Home_Goods":  {"base_price": 55,  "price_std": 15, "base_units": 12},
    "Toys":        {"base_price": 25,  "price_std": 8,  "base_units": 10},
    "Beauty":      {"base_price": 20,  "price_std": 6,  "base_units": 15},
}

rows = []
for d in dates:
    day_of_week = d.dayofweek  # 0=Mon
    month = d.month
    is_weekend = day_of_week >= 5
    # Seasonal multiplier: holiday bump in Nov/Dec, summer bump in Jun/Jul
    seasonal_mult = 1.0
    if month in (11, 12):
        seasonal_mult = 1.45
    elif month in (6, 7):
        seasonal_mult = 1.15
    elif month in (1, 2):
        seasonal_mult = 0.85

    weekend_mult = 1.25 if is_weekend else 1.0

    # Random promotion days (~12% of days per store/category)
    for store in stores:
        store_mult = store_base_traffic[store] / 200.0
        for cat, props in categories.items():
            promo = np.random.rand() < 0.12
            promo_mult = 1.35 if promo else 1.0

            units = np.random.poisson(
                lam=max(props["base_units"] * seasonal_mult * weekend_mult * store_mult * promo_mult, 1)
            )
            price = max(np.random.normal(props["base_price"], props["price_std"]), 1)
            if promo:
                price = price * 0.85  # discount during promo

            revenue = round(units * price, 2)

            rows.append({
                "date": d,
                "store": store,
                "category": cat,
                "units_sold": units,
                "unit_price": round(price, 2),
                "promotion": int(promo),
                "day_of_week": day_of_week,
                "is_weekend": int(is_weekend),
                "month": month,
                "revenue": revenue,
            })

df = pd.DataFrame(rows)

# Introduce a small amount of missingness/noise to simulate real-world messiness
mask = np.random.rand(len(df)) < 0.01
df.loc[mask, "unit_price"] = np.nan

out_path = "/home/claude/retail-data-project/data/retail_sales.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
print(df.head())
