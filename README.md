# 🛍️ Real-World Data Project — Retail Sales Analysis & Revenue Prediction

An end-to-end applied data science project built on a domain-specific **retail sales dataset**. The project covers data generation/cleaning, exploratory data analysis (EDA), visualization, and a machine learning model that predicts transaction revenue.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| Domain | Retail |
| Dataset size | 21,930 transactions |
| Time span | Jan 1, 2024 – Dec 31, 2025 (2 years) |
| Stores | 5 (`Store_A`–`Store_E`) |
| Product categories | 6 (Electronics, Groceries, Clothing, Home Goods, Toys, Beauty) |
| Task type | Regression (predict transaction revenue) |
| Best model | Random Forest Regressor |
| Best model R² | **0.997** |

---

## 🗂️ Project Structure

```
retail-data-project/
├── data/
│   └── retail_sales.csv           # Raw generated dataset (21,930 rows)
├── notebooks/
│   ├── 01_generate_data.py        # Synthetic data generation with seasonality & promos
│   └── 02_analysis_and_model.py   # Cleaning, EDA, charts, model training
├── outputs/
│   ├── charts/                    # 7 saved PNG visualizations
│   └── reports/
│       ├── category_summary.csv
│       ├── store_summary.csv
│       └── summary.json
└── README.md
```

---

## 🧹 Data Cleaning

| Step | Result |
|---|---|
| Missing `unit_price` values found | 182 |
| Missing value treatment | Filled with category-level median price |
| Missing values after cleaning | 0 |
| Derived column added | `revenue = units_sold × unit_price` |

---

## 📊 Exploratory Data Analysis

### 1. Monthly Revenue Trend
Shows clear seasonality — a strong holiday-season lift in November/December and a smaller summer bump in June/July.

![Monthly Revenue Trend](outputs/charts/01_monthly_revenue_trend.png)

### 2. Revenue by Product Category

| Category | Total Revenue | Avg Units Sold/Txn | Avg Unit Price | Transactions |
|---|---:|---:|---:|---:|
| Electronics | $4,135,157.93 | 9.61 | $118.10 | 3,655 |
| Home Goods | $2,846,089.64 | 14.38 | $54.24 | 3,655 |
| Clothing | $2,677,483.83 | 21.51 | $34.29 | 3,655 |
| Groceries | $2,575,146.44 | 48.21 | $14.75 | 3,655 |
| Beauty | $1,296,861.19 | 18.13 | $19.71 | 3,655 |
| Toys | $1,057,461.97 | 11.91 | $24.41 | 3,655 |

![Revenue by Category](outputs/charts/02_revenue_by_category.png)

Electronics leads in total revenue despite modest unit volume, driven by a high average price point — while Groceries sells the most units per transaction but at a low price point.

### 3. Revenue by Store

| Store | Total Revenue | Avg Daily Revenue | Transactions |
|---|---:|---:|---:|
| Store_C | $3,776,839.37 | $861.11 | 4,386 |
| Store_A | $3,224,583.67 | $735.20 | 4,386 |
| Store_E | $2,947,772.27 | $672.09 | 4,386 |
| Store_B | $2,618,951.57 | $597.12 | 4,386 |
| Store_D | $2,020,054.12 | $460.57 | 4,386 |

![Revenue by Store](outputs/charts/03_revenue_by_store.png)

Store_C is the top performer, generating ~87% more revenue than the lowest-performing Store_D — a strong candidate for a follow-up "why is Store_D underperforming" investigation.

### 4. Promotion & Weekend Effects

![Promotion and Weekend Effect](outputs/charts/04_promo_weekend_effect.png)

Both promotions and weekends independently lift average transaction revenue, with the combination of "weekend + promotion" producing the highest average basket value.

### 5. Feature Correlation

![Correlation Heatmap](outputs/charts/05_correlation_heatmap.png)

`units_sold` and `unit_price` are, unsurprisingly, the strongest drivers of `revenue`. Calendar features (day of week, month) show weaker but non-trivial correlation, consistent with the seasonality seen in Chart 1.

---

## 🤖 Predictive Modeling

**Goal:** Predict per-transaction `revenue` from operational features (units sold, unit price, promotion flag, store, category, day of week, weekend flag, month).

**Approach:** 80/20 train-test split, one-hot encoding for categorical features, benchmarked two models.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | $142.73 | $220.74 | 0.766 |
| **Random Forest** ⭐ | **$7.52** | **$24.98** | **0.997** |

The Random Forest model dramatically outperforms the linear baseline, capturing non-linear interactions between promotions, category, and price that a linear model misses.

### Actual vs. Predicted Revenue (Best Model)

![Actual vs Predicted](outputs/charts/06_actual_vs_predicted.png)

Predictions cluster tightly along the diagonal "perfect prediction" line, confirming strong model fit with no major bias at high or low revenue values.

### Feature Importance

![Feature Importance](outputs/charts/07_feature_importance.png)

As expected, `units_sold` and `unit_price` dominate importance, with `category` and `promotion` contributing secondary but meaningful signal.

---

## ✅ Key Findings & Conclusions

1. **Seasonality is strong and predictable** — Nov/Dec holiday season drives the largest revenue spike of the year, useful for inventory and staffing planning.
2. **Electronics is the highest-value category** by revenue despite lower unit volume — pricing strategy matters more than volume here.
3. **Store performance varies significantly** — Store_C outperforms Store_D by ~87%, warranting operational review.
4. **Promotions work** — they reliably increase average transaction revenue, especially when paired with weekend traffic.
5. **Revenue is highly predictable** from operational features — a Random Forest model achieves R² = 0.997, meaning it could realistically support demand forecasting or automated pricing checks.

---

## 🔁 Reproducing This Project

```bash
pip install pandas numpy matplotlib scikit-learn

python notebooks/01_generate_data.py        # creates data/retail_sales.csv
python notebooks/02_analysis_and_model.py   # creates all charts + reports
```

---

## 🚀 Possible Extensions

- Add time-series forecasting (Prophet / ARIMA) for future revenue prediction
- Incorporate external data (weather, local events, competitor pricing)
- Deploy the Random Forest model behind a simple API for real-time revenue estimation
- Build an interactive dashboard (Streamlit/Plotly Dash) on top of `outputs/reports/`

---

*This project uses a synthetically generated dataset built to mimic realistic retail sales patterns (seasonality, promotions, weekend effects, and natural data messiness) for applied learning purposes.*
