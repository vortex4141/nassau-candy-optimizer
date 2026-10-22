# 🍬 Nassau Candy: Factory Reallocation & Shipping Optimizer

An interactive Streamlit dashboard that predicts shipping lead times for Nassau Candy Distributor and recommends which factory should produce each product to ship faster without hurting profit.

## Problem

Each product is made at a fixed factory, but the distance from factory to customer region drives shipping lead time. This project answers one question: **would reassigning a product to a different factory reduce lead time while keeping profit margins healthy?**

## What it does

- **Predicts lead time** with three ML models (Linear Regression, Random Forest, Gradient Boosting) and automatically selects the best one by R².
- **Simulates every factory** for a chosen product, region and ship mode, and scores each option on speed and profit with an adjustable priority slider (α).
- **Compares current vs recommended** assignment in a what-if view, with a waterfall chart and historical trend.
- **Runs bulk recommendations** across all 15 products and exports them to CSV.
- **Flags risk**: high lead time + low margin orders, a factory × region heatmap, congested region-product pairs, and profit alerts for risky reassignments.
- **Clusters routes** (K-Means) on distance, lead time, margin and cost to separate efficient routes from problem routes.

## Dashboard tabs

| Tab | Purpose |
|---|---|
| 📊 Overview & KPIs | Lead time, margin and volume by factory, division, region and ship mode |
| 🤖 Model Performance | RMSE / MAE / R² comparison, feature importance, route clusters |
| 🏭 Factory Simulator | Predicted lead time and margin trade-off for each factory |
| 🔄 What-If Scenario | Current vs recommended factory side by side |
| 🏆 Recommendations | Bulk reassignment suggestions with CSV export |
| ⚠️ Risk & Impact | High-risk orders, heatmap, treemap, profit alerts |

## Method

1. **Data prep**: parse order and ship dates, compute lead time, map products to factories, and calculate profit margin.
2. **Feature engineering**: haversine distance (miles) from each factory to each region centroid.
3. **Modeling**: label-encode categoricals, median-impute numerics, train/test split, compare three regressors on RMSE, MAE and R².
4. **Optimization**: for each candidate factory, predict lead time and combine it with historical profit margin into a composite score:
   `score = (1 − α) × speed score + α × profit score`
5. **Recommendation**: the highest-scoring factory per product, region and ship mode.

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · Plotly

## Run locally

```bash
git clone https://github.com/vortex4141/nassau-candy-optimizer.git
cd nassau-candy-optimizer
pip install -r requirements.txt
streamlit run nassau_candy_optimizer.py
```

Keep `nassau dataset.xlsx` in the same folder as the script.

## Project structure

```
├── nassau_candy_optimizer.py   # Streamlit app (data, models, simulation, dashboard)
├── nassau dataset.xlsx         # Order data
├── requirements.txt
└── README.md
```

## Limitations

- Region distances use region centroids, not exact customer addresses.
- Profit margin per factory is based on historical orders, so unseen product-factory pairs fall back to the overall average.
- Recommendations are model predictions and should be validated before any real reassignment.

## Author

**Tushar Sanap**: MBA (Business Analytics) candidate, Pune, India.
