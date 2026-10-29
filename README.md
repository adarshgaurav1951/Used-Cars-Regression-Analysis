# Used Car Price Prediction & Regression Valuation Engine

An end-to-end Machine Learning regression project and interactive valuation application developed according to the specifications in [prd.md](prd.md). The system estimates pre-owned car fair market prices in Indian Rupees (₹), analyzes depreciation curves, and provides strategic pricing recommendations for buyers, sellers, and dealership appraisers.

---

## 🌟 Key Features

1. **Exploratory Data Analysis & Depreciation Modeling:**
   - 10-section top-to-bottom pipeline in [used_car_price.ipynb](used_car_price.ipynb).
   - In-depth analysis of vehicle age, odometer mileage, fuel efficiency, transmission type, and ownership tier.
   - Target log-normalization (`np.log1p`) to eliminate heavy price skew and stabilize regression residuals.

2. **Leak-Free Preprocessing Architecture:**
   - Strict separation between training (80%) and test (20%) data.
   - Encapsulated scikit-learn `ColumnTransformer` with `StandardScaler` for numeric variables and `OneHotEncoder(drop='first', handle_unknown='ignore')` for categoricals.
   - Pipeline fitting performed strictly on training data.

3. **Multi-Model Benchmark:**
   - **Linear Regression & Ridge Regression:** Benchmark models achieving test $R^2 \approx 0.670$ in Rupees ($R^2 \approx 0.775$ on log scale) and lowest MAE ($\approx ₹1.47$ Lakhs).
   - **Gradient Boosting Regressor & Random Forest:** Capturing non-linear thresholds and feature importances.

4. **Interactive Valuation Web Application ([app.py](app.py)):**
   - **Persona-Driven Strategies:** Tailored pricing advice for **Rahul (Buyer)**, **Ananya (Seller)**, and **Vikram (Dealership Manager)**.
   - **What-If Sensitivity Simulator:** Real-time calculation of value delta for transmission shifts, mileage increases, and ownership transitions.
   - **Model Transparency:** Displays live PRD benchmark metrics and top feature importance rankings.

---

## 📊 Benchmark Model Performance

| Model | Train $R^2$ (₹) | Test $R^2$ (₹) | Test $R^2$ (Log Scale) | Test MAE (₹) | Test RMSE (₹) |
|---|---|---|---|---|---|
| **Ridge Regression (Recommended)** | 0.6822 | **0.6696** | **0.7740** | **₹1,47,720** | ₹3,26,233 |
| **Linear Regression (Baseline)** | 0.7035 | **0.6698** | **0.7748** | **₹1,47,370** | ₹3,26,119 |
| **Gradient Boosting** | 0.6885 | 0.6524 | 0.7730 | ₹1,48,042 | ₹3,34,620 |
| **Random Forest** | 0.8986 | 0.6151 | 0.7558 | ₹1,52,322 | ₹3,52,123 |

*Evaluated on 716 unseen test listings with predictions mapped back to Indian Rupees.*

---

## 📁 Repository Structure

```
Used Cars Regression Analysis/
├── car_data.csv               # Clean source dataset (CarDekho pre-owned listings)
├── used_car_price.ipynb       # 10-section analysis notebook (unmodified & fully executed)
├── prd.md                     # Complete Product Requirements Document
├── idea.md                    # Project blueprint & roadmap
├── AGENTS.md                  # Agent rules and operational guidelines
├── app.py                     # Streamlit valuation web application
├── README.md                  # Project overview and usage guide
├── models/
│   ├── ridge_pipeline.joblib             # Serialized Ridge regression pipeline
│   ├── gradient_boosting_pipeline.joblib # Serialized Gradient Boosting pipeline
│   └── metadata.json                     # Feature schemas, categories, and metrics
└── venv/                      # Local Python virtual environment
```

---

## 🚀 Quick Start Guide

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Run the Valuation Web Application
```bash
./venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive price estimator.

### 3. View the Notebook
Open [used_car_price.ipynb](used_car_price.ipynb) in Antigravity or VS Code using the `Python (used-cars venv)` kernel to explore the 10 analysis sections, visualizations, and statistical findings.
