import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Used Car Price Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern UI aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1d3557, #457b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 5px solid #1d3557;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .price-display {
        font-size: 2.4rem;
        font-weight: 900;
        color: #1d3557;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-bottom: 0.5rem;
    }
    .persona-box {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format Indian Rupees
def format_inr(number):
    if number >= 10000000:
        return f"₹{number / 10000000:.2f} Crore"
    elif number >= 100000:
        return f"₹{number / 100000:.2f} Lakhs"
    else:
        return f"₹{number:,.0f}"

# Load models and metadata with caching
@st.cache_resource
def load_assets():
    model_path = "models/ridge_pipeline.joblib"
    gb_model_path = "models/gradient_boosting_pipeline.joblib"
    meta_path = "models/metadata.json"
    
    ridge_model = joblib.load(model_path)
    gb_model = joblib.load(gb_model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return ridge_model, gb_model, metadata

try:
    ridge_pipeline, gb_pipeline, meta = load_assets()
except Exception as e:
    st.error(f"Error loading models or metadata: {e}. Please ensure models are trained and saved in 'models/'.")
    st.stop()

# Application Header
st.markdown('<div class="main-header">Used Car Fair-Market Valuation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven pre-owned vehicle pricing powered by regularized regression & gradient boosting pipelines</div>', unsafe_allow_html=True)

# Sidebar: Persona & Model Configuration
st.sidebar.header("🎯 Valuation Persona")
persona = st.sidebar.radio(
    "Select Your Perspective:",
    options=[
        "Rahul (Private Buyer)",
        "Ananya (Private Seller)",
        "Vikram (Dealership Manager)"
    ],
    help="Adapts recommendations, margin buffers, and pricing strategies based on PRD User Personas."
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Estimation Engine")
selected_model_type = st.sidebar.selectbox(
    "Active Regression Model:",
    ["Ridge Regression (Best Generalization & Lowest MAE)", "Gradient Boosting (Non-linear Tree Ensemble)"],
    index=0
)
active_pipeline = ridge_pipeline if "Ridge" in selected_model_type else gb_pipeline

# Layout: 2 Columns (Inputs vs Output & Strategy)
col_input, col_output = st.columns([1.1, 1.3], gap="large")

with col_input:
    st.subheader("📋 Vehicle Specifications")
    
    brand = st.selectbox(
        "Car Brand / Make",
        options=meta["common_brands"],
        index=meta["common_brands"].index("Maruti") if "Maruti" in meta["common_brands"] else 0
    )
    
    year_col, km_col = st.columns(2)
    with year_col:
        year = st.slider("Manufacturing Year", min_value=1995, max_value=meta["reference_year"], value=2015, step=1)
    with km_col:
        km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=600000, value=55000, step=2500)
        
    fuel = st.selectbox("Fuel Type", options=meta["fuels"], index=meta["fuels"].index("Petrol") if "Petrol" in meta["fuels"] else 0)
    transmission = st.radio("Transmission", options=meta["transmissions"], horizontal=True, index=0)
    
    seller_col, owner_col = st.columns(2)
    with seller_col:
        seller_type = st.selectbox("Seller Category", options=meta["seller_types"], index=0)
    with owner_col:
        owner = st.selectbox("Ownership History", options=meta["owners"], index=0)

# Feature engineering for inference
car_age = meta["reference_year"] - year
km_per_year = km_driven / (car_age + 1)

input_df = pd.DataFrame([{
    "car_age": car_age,
    "km_driven": km_driven,
    "km_per_year": km_per_year,
    "brand": brand,
    "fuel": fuel,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner
}])

# Compute Predictions
log_pred = active_pipeline.predict(input_df)[0]
predicted_price = float(np.expm1(log_pred))

# Benchmark MAE from PRD
active_mae = meta["metrics"]["Ridge"]["mae"] if "Ridge" in selected_model_type else meta["metrics"]["GradientBoosting"]["mae"]
lower_bound = max(20000.0, predicted_price - active_mae)
upper_bound = predicted_price + active_mae

with col_output:
    st.subheader("💡 Fair Market Valuation")
    
    # Hero Metric Card
    st.markdown(f"""
    <div class="metric-card">
        <span class="confidence-badge">R² Score: ~0.670 | Test MAE: ±₹{active_mae:,.0f}</span>
        <div style="font-size: 0.95rem; color: #6c757d; font-weight: 600;">ESTIMATED FAIR SELLING PRICE</div>
        <div class="price-display">{format_inr(predicted_price)}</div>
        <div style="font-size: 0.95rem; color: #495057;">
            Expected Fair Market Range: <b>{format_inr(lower_bound)}</b> — <b>{format_inr(upper_bound)}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Persona-Driven Practical Recommendations
    if "Buyer" in persona:
        st.markdown(f"""
        <div class="persona-box" style="background-color: #f0fdf4; border-color: #bbf7d0;">
            <b style="color: #166534; font-size: 1.05rem;">🎯 Buyer's Negotiation Playbook (Rahul)</b>
            <ul style="color: #14532d; font-size: 0.92rem; margin-top: 0.5rem; padding-left: 1.2rem;">
                <li><b>Fair Offer Target:</b> Begin counter-offers around <b>{format_inr(lower_bound)}</b>.</li>
                <li><b>Walk-Away Ceiling:</b> Avoid paying above <b>{format_inr(predicted_price * 1.05)}</b> unless complete service logs and recent tire replacements are documented.</li>
                <li><b>Inspection Priority:</b> With <b>{km_per_year:,.0f} km/year</b> annual usage rate, request brake pad and suspension inspection.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif "Seller" in persona:
        listing_price = predicted_price * 1.07
        st.markdown(f"""
        <div class="persona-box" style="background-color: #eff6ff; border-color: #bfdbfe;">
            <b style="color: #1e40af; font-size: 1.05rem;">💼 Seller's Listing Strategy (Ananya)</b>
            <ul style="color: #1e3a8a; font-size: 0.92rem; margin-top: 0.5rem; padding-left: 1.2rem;">
                <li><b>Recommended Listing Price:</b> <b>{format_inr(listing_price)}</b> (includes 7% cushion for buyer negotiation).</li>
                <li><b>Minimum Acceptable Price:</b> Settle no lower than <b>{format_inr(lower_bound)}</b> for a fast transaction.</li>
                <li><b>Instant Trade-in Benchmark:</b> Dealership trade-in offers typically range between <b>{format_inr(predicted_price * 0.85)}</b> and <b>{format_inr(predicted_price * 0.90)}</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Vikram (Dealership Manager)
        trade_in_max = predicted_price * 0.82
        showroom_retail = predicted_price * 1.10
        dealer_margin = showroom_retail - trade_in_max
        st.markdown(f"""
        <div class="persona-box" style="background-color: #fefce8; border-color: #fef08a;">
            <b style="color: #854d0e; font-size: 1.05rem;">🏢 Dealership Appraisal & Spread (Vikram)</b>
            <ul style="color: #713f12; font-size: 0.92rem; margin-top: 0.5rem; padding-left: 1.2rem;">
                <li><b>Max Trade-in Acquisition Offer:</b> <b>{format_inr(trade_in_max)}</b> (protects refurbishing budget).</li>
                <li><b>Showroom Retail Target:</b> <b>{format_inr(showroom_retail)}</b> (certified warranty pricing).</li>
                <li><b>Projected Gross Margin Spread:</b> ~<b>{format_inr(dealer_margin)}</b> per unit turnaround.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Section: "What-If" Sensitivity Simulator & Model Insights
tab_sim, tab_insights, tab_metrics, tab_workflow = st.tabs([
    "⚡ What-If Sensitivity Simulator", 
    "📊 Key Price Drivers", 
    "📈 PRD Benchmark Metrics",
    "📖 End-to-End Workflow & Architecture"
])

with tab_sim:
    st.markdown("##### Discover how vehicle modifications alter fair market value:")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    # 1. Transmission impact
    alt_trans = "Automatic" if transmission == "Manual" else "Manual"
    sim_trans_df = input_df.copy()
    sim_trans_df["transmission"] = alt_trans
    alt_trans_price = float(np.expm1(active_pipeline.predict(sim_trans_df)[0]))
    trans_diff = alt_trans_price - predicted_price
    
    with sim_col1:
        st.metric(
            label=f"Switch to {alt_trans} Transmission",
            value=format_inr(alt_trans_price),
            delta=f"{trans_diff:+,.0f} ₹"
        )
        
    # 2. Additional 25,000 km
    sim_km_df = input_df.copy()
    sim_km_df["km_driven"] = km_driven + 25000
    sim_km_df["km_per_year"] = (km_driven + 25000) / (car_age + 1)
    alt_km_price = float(np.expm1(active_pipeline.predict(sim_km_df)[0]))
    km_diff = alt_km_price - predicted_price
    
    with sim_col2:
        st.metric(
            label="+25,000 km Higher Odometer",
            value=format_inr(alt_km_price),
            delta=f"{km_diff:+,.0f} ₹"
        )
        
    # 3. Subsequent ownership change
    alt_owner = "Second Owner" if owner == "First Owner" else "Third Owner"
    sim_owner_df = input_df.copy()
    sim_owner_df["owner"] = alt_owner
    alt_owner_price = float(np.expm1(active_pipeline.predict(sim_owner_df)[0]))
    owner_diff = alt_owner_price - predicted_price
    
    with sim_col3:
        st.metric(
            label=f"Down-tier to {alt_owner}",
            value=format_inr(alt_owner_price),
            delta=f"{owner_diff:+,.0f} ₹"
        )

with tab_insights:
    st.markdown("##### Top Feature Importances (from Gradient Boosting Regressor)")
    fi = meta["feature_importances"]
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    fi_df = pd.DataFrame(sorted_fi, columns=["Feature", "Importance Score"])
    st.bar_chart(fi_df.set_index("Feature"))

with tab_metrics:
    st.markdown("##### PRD Regression Benchmark Summary")
    metrics_display = pd.DataFrame([
        {
            "Model": "Ridge Regression",
            "Test R² (Rupees)": meta["metrics"]["Ridge"]["r2"],
            "Test MAE (₹)": f"₹{meta['metrics']['Ridge']['mae']:,.2f}",
            "Test RMSE (₹)": f"₹{meta['metrics']['Ridge']['rmse']:,.2f}",
            "Leakage Guard": "Pipeline + ColumnTransformer",
            "Verdict": "Primary Recommended (Lowest MAE)"
        },
        {
            "Model": "Gradient Boosting",
            "Test R² (Rupees)": meta["metrics"]["GradientBoosting"]["r2"],
            "Test MAE (₹)": f"₹{meta['metrics']['GradientBoosting']['mae']:,.2f}",
            "Test RMSE (₹)": f"₹{meta['metrics']['GradientBoosting']['rmse']:,.2f}",
            "Leakage Guard": "Pipeline + ColumnTransformer",
            "Verdict": "Top Non-Linear Ensemble"
        }
    ])
    st.table(metrics_display)
    st.caption("Training sample size: 2,860 cars | Test holdout: 716 cars | Target transformed via np.log1p for variance stabilization.")

with tab_workflow:
    st.markdown("### 🗺️ End-to-End Project Workflow & Architecture")
    
    # Display the generated workflow infographic
    if os.path.exists("assets/ml_project_workflow.jpg"):
        st.image("assets/ml_project_workflow.jpg", caption="End-to-End Machine Learning Used Car Price Prediction Pipeline Architecture", use_container_width=True)
    
    st.markdown("""
    #### 🎯 1. Main Goal of the Project
    Buyers and sellers in the used car market face severe information asymmetry. The primary objective is to:
    - Eliminate guesswork by estimating fair market prices in Indian Rupees (₹).
    - Prevent model data leakage using encapsulated scikit-learn pipelines.
    - Identify and quantify the strongest drivers of car depreciation (age, transmission, brand tier).
    
    ---
    
    #### 🏗️ 2. How the Project Was Built from Start to End
    
    1. **Data Ingestion & Integrity Validation:**  
       Raw data from CarDekho (`car_data.csv`) loaded into Pandas. The source CSV remains immutable throughout.
    
    2. **Data Cleaning & Anomaly Filtering:**  
       763 duplicate rows were eliminated (reducing rows from 4,340 to 3,577). A data entry error (a 2014 car with 1 km driven) was filtered.
    
    3. **Exploratory Data Analysis (EDA):**  
       Visualized price distribution skewness, depreciation curves against manufacturing year, mileage decay, and box plots comparing fuel and transmission types.
    
    4. **Domain Feature Engineering:**  
       Created `car_age` ($2020 - year$), consolidated rare brands into `Other`, engineered `km_per_year` (annual usage intensity), and normalized target prices via `np.log1p`.
    
    5. **Leak-Free Preprocessing Architecture:**  
       Partitioned data (80% train / 20% test). Standardized numerics using `StandardScaler` and encoded categoricals with `OneHotEncoder(drop='first', handle_unknown='ignore')` inside a `ColumnTransformer`.
    
    6. **Multi-Model Training & Benchmarking:**  
       Trained 4 distinct algorithms (Linear Regression, Ridge, Random Forest, Gradient Boosting) using strictly isolated scikit-learn pipelines.
    
    7. **Evaluation & Explainability:**  
       Inverted log predictions to Rupees using `np.expm1()`. Benchmarked on test data ($R^2$, MAE, RMSE) and extracted feature importances.
    
    8. **Model Serialization & Interactive App:**  
       Exported production pipelines using `joblib` and created this interactive Streamlit valuation engine with persona support.
    
    ---
    
    #### 🛠️ 3. Complete Technical Stack Breakdown
    
    | Stage | Library Used | Functions & Classes | Purpose & Logic |
    |---|---|---|---|
    | **Data Handling** | `pandas` | `pd.read_csv`, `df.drop_duplicates()`, `df.isnull().sum()`, `df.apply()` | Tabular ingestion, duplicate removal, brand extraction, summary stats. |
    | **Math & Transforms** | `numpy` | `np.log1p()`, `np.expm1()` | Natural log-transform to compress skewed prices; exponentiation back to ₹. |
    | **Visualizations** | `matplotlib`, `seaborn` | `plt.subplots()`, `sns.histplot()`, `sns.scatterplot()`, `sns.boxplot()`, `sns.heatmap()`, `sns.barplot()` | Distribution charts, scatter correlations, box plots, heatmap, feature importance plot. |
    | **Preprocessing** | `scikit-learn` | `StandardScaler()`, `OneHotEncoder(drop='first')`, `ColumnTransformer()` | Normalizes numerical scales and creates dummy variables without leakage. |
    | **Workflow Guard** | `scikit-learn` | `Pipeline()` | Chains preprocessing and model estimation; guarantees test data is never leaked. |
    | **Validation** | `scikit-learn` | `train_test_split(random_state=42)`, `KFold()`, `cross_val_score()` | Holdout evaluation (80/20) and 5-fold cross-validation. |
    | **Linear ML Models** | `scikit-learn` | `LinearRegression()`, `Ridge(alpha=1.0)` | Unregularized OLS baseline and L2 regularized regression (Top performer: MAE ~₹1.47 Lakhs). |
    | **Ensemble ML Models**| `scikit-learn` | `RandomForestRegressor()`, `GradientBoostingRegressor()` | Bagged decision trees and boosted residual trees for non-linear thresholds. |
    | **Evaluation Metrics**| `scikit-learn` | `r2_score()`, `mean_absolute_error()`, `root_mean_squared_error()` | Variance explained ($R^2$) and typical absolute price error in ₹ (MAE). |
    | **Model Persistence** | `joblib` | `joblib.dump()`, `joblib.load()` | Serializing trained pipelines to disk for real-time web inference. |
    | **Web Application** | `streamlit` | `st.set_page_config()`, `st.sidebar`, `st.metric()`, `st.tabs()`, `st.image()` | Interactive UI with real-time prediction and persona-specific strategy cards. |
    """)

