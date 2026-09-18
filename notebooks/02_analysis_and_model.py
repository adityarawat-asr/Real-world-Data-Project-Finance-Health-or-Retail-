"""
02_analysis_and_model.py
End-to-end analysis: cleaning, EDA visualizations, and a revenue prediction model.
Outputs charts to outputs/charts/ and a metrics/summary json to outputs/reports/.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
PALETTE = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#9333ea", "#0891b2"]

DATA_PATH = "/home/claude/retail-data-project/data/retail_sales.csv"
CHART_DIR = "/home/claude/retail-data-project/outputs/charts"
REPORT_DIR = "/home/claude/retail-data-project/outputs/reports"

# ---------------- 1. Load & clean ----------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
missing_before = df.isna().sum().sum()
df["unit_price"] = df.groupby("category")["unit_price"].transform(lambda s: s.fillna(s.median()))
df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
missing_after = df.isna().sum().sum()

summary = {}
summary["n_rows"] = int(len(df))
summary["n_stores"] = int(df["store"].nunique())
summary["n_categories"] = int(df["category"].nunique())
summary["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
summary["missing_values_before_cleaning"] = int(missing_before)
summary["missing_values_after_cleaning"] = int(missing_after)
summary["total_revenue"] = float(df["revenue"].sum())
summary["avg_daily_revenue"] = float(df.groupby("date")["revenue"].sum().mean())

# ---------------- 2. EDA charts ----------------

# Chart 1: Monthly revenue trend
monthly = df.groupby(pd.Grouper(key="date", freq="ME"))["revenue"].sum()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(monthly.index, monthly.values, color=PALETTE[0], linewidth=2, marker="o", markersize=3)
ax.set_title("Monthly Revenue Trend (2024–2025)", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/01_monthly_revenue_trend.png")
plt.close(fig)

# Chart 2: Revenue by category (bar)
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(cat_rev.index, cat_rev.values, color=PALETTE)
ax.set_title("Total Revenue by Product Category", fontweight="bold")
ax.set_ylabel("Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
plt.xticks(rotation=20, ha="right")
for b in bars:
    ax.annotate(f"${b.get_height()/1000:.0f}K", (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/02_revenue_by_category.png")
plt.close(fig)

# Chart 3: Revenue by store
store_rev = df.groupby("store")["revenue"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(store_rev.index[::-1], store_rev.values[::-1], color=PALETTE[1])
ax.set_title("Total Revenue by Store", fontweight="bold")
ax.set_xlabel("Revenue ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/03_revenue_by_store.png")
plt.close(fig)

# Chart 4: Weekend vs weekday + promotion effect (grouped bar)
grp = df.groupby(["is_weekend", "promotion"])["revenue"].mean().unstack()
grp.index = ["Weekday", "Weekend"]
grp.columns = ["No Promotion", "Promotion"]
fig, ax = plt.subplots(figsize=(7, 4.5))
grp.plot(kind="bar", ax=ax, color=[PALETTE[2], PALETTE[3]])
ax.set_title("Avg Revenue per Transaction: Promotion & Weekend Effect", fontweight="bold")
ax.set_ylabel("Avg Revenue ($)")
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/04_promo_weekend_effect.png")
plt.close(fig)

# Chart 5: Correlation heatmap of numeric features
num_cols = ["units_sold", "unit_price", "promotion", "day_of_week", "is_weekend", "month", "revenue"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
ax.set_yticks(range(len(num_cols)), num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                 color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=8)
ax.set_title("Correlation Heatmap of Numeric Features", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/05_correlation_heatmap.png")
plt.close(fig)

# ---------------- 3. Predictive model: forecast revenue per transaction ----------------
features_num = ["units_sold", "unit_price", "promotion", "day_of_week", "is_weekend", "month"]
features_cat = ["store", "category"]
target = "revenue"

X = df[features_num + features_cat]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
}

model_results = {}
best_name, best_r2, best_pipe = None, -np.inf, None
for name, mdl in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", mdl)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    model_results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    if r2 > best_r2:
        best_name, best_r2, best_pipe = name, r2, pipe

summary["model_results"] = model_results
summary["best_model"] = best_name

# Chart 6: Actual vs predicted (best model)
best_preds = best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_test, best_preds, alpha=0.25, s=12, color=PALETTE[0])
lims = [0, max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, color=PALETTE[3], linestyle="--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Revenue ($)")
ax.set_ylabel("Predicted Revenue ($)")
ax.set_title(f"Actual vs Predicted Revenue — {best_name} (R²={best_r2:.3f})", fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(f"{CHART_DIR}/06_actual_vs_predicted.png")
plt.close(fig)

# Chart 7: Feature importance (only meaningful for Random Forest)
if best_name == "Random Forest":
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(features_cat))
    all_feature_names = cat_feature_names + features_num
    importances = best_pipe.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_series.index[::-1], imp_series.values[::-1], color=PALETTE[4])
    ax.set_title("Top 12 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/07_feature_importance.png")
    plt.close(fig)

# ---------------- 4. Save summary tables ----------------
cat_summary = df.groupby("category").agg(
    total_revenue=("revenue", "sum"),
    avg_units_sold=("units_sold", "mean"),
    avg_unit_price=("unit_price", "mean"),
    transactions=("revenue", "count"),
).round(2).sort_values("total_revenue", ascending=False)
cat_summary.to_csv(f"{REPORT_DIR}/category_summary.csv")

store_summary = df.groupby("store").agg(
    total_revenue=("revenue", "sum"),
    avg_daily_revenue=("revenue", "mean"),
    transactions=("revenue", "count"),
).round(2).sort_values("total_revenue", ascending=False)
store_summary.to_csv(f"{REPORT_DIR}/store_summary.csv")

with open(f"{REPORT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Analysis complete.")
print(json.dumps(summary, indent=2))
print("\nCategory summary:\n", cat_summary)
print("\nStore summary:\n", store_summary)
