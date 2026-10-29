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

# Read and sync URL Query Parameters for persistence
qp = st.query_params
def_brand = qp.get("brand", "Maruti")
if def_brand not in meta["common_brands"]:
    def_brand = "Maruti"

try:
    def_year = int(qp.get("year", 2012))
except (ValueError, TypeError):
    def_year = 2012

try:
    def_km = int(qp.get("km", 50000))
except (ValueError, TypeError):
    def_km = 50000

def_fuel = qp.get("fuel", "Petrol")
if def_fuel not in meta["fuels"]:
    def_fuel = "Petrol"

def_trans = qp.get("trans", "Automatic")
if def_trans not in meta["transmissions"]:
    def_trans = "Automatic"

def_seller = qp.get("seller", "Dealer")
if def_seller not in meta["seller_types"]:
    def_seller = "Dealer"

def_owner = qp.get("owner", "First Owner")
if def_owner not in meta["owners"]:
    def_owner = "First Owner"

persona_options = [
    "Rahul (Private Buyer)",
    "Ananya (Private Seller)",
    "Vikram (Dealership Manager)"
]
def_persona = qp.get("persona", "Rahul (Private Buyer)")
if def_persona not in persona_options:
    def_persona = "Rahul (Private Buyer)"

# Sidebar: Persona & Model Configuration
st.sidebar.header("🎯 Valuation Persona")
persona = st.sidebar.radio(
    "Select Your Perspective:",
    options=persona_options,
    index=persona_options.index(def_persona),
    help="Adapts recommendations, margin buffers, and pricing strategies based on PRD User Personas."
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Estimation Engine")
selected_model_type = st.sidebar.selectbox(
    "Active Regression Model:",
    ["Ridge Regression (Best Generalization & Lowest MAE)", "Gradient Boosting (Non-Linear Tree Ensemble)"],
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
        index=meta["common_brands"].index(def_brand)
    )
    
    year_col, km_col = st.columns(2)
    with year_col:
        year = st.slider("Manufacturing Year", min_value=1995, max_value=meta["reference_year"], value=def_year, step=1)
    with km_col:
        km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=600000, value=def_km, step=2500)
        
    fuel = st.selectbox("Fuel Type", options=meta["fuels"], index=meta["fuels"].index(def_fuel))
    transmission = st.radio("Transmission", options=meta["transmissions"], horizontal=True, index=meta["transmissions"].index(def_trans))
    
    seller_col, owner_col = st.columns(2)
    with seller_col:
        seller_type = st.selectbox("Seller Category", options=meta["seller_types"], index=meta["seller_types"].index(def_seller))
    with owner_col:
        owner = st.selectbox("Ownership History", options=meta["owners"], index=meta["owners"].index(def_owner))

# Automatically update URL parameters so any bookmark or link retains exact user state
st.query_params["brand"] = brand
st.query_params["year"] = str(year)
st.query_params["km"] = str(km_driven)
st.query_params["fuel"] = fuel
st.query_params["trans"] = transmission
st.query_params["seller"] = seller_type
st.query_params["owner"] = owner
st.query_params["persona"] = persona

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
    st.markdown("### 🗺️ Visual Architecture & Project Blueprint")
    st.markdown("Explore the complete end-to-end engineering, execution roadmap, and technology stack powering the Used Car Valuation Engine.")
    
    wf_subtab1, wf_subtab2, wf_subtab3 = st.tabs([
        "🎯 1. Main Goal of this Project",
        "🏗️ 2. 5 Stages & 10 Sections Workflow",
        "🛠️ 3. Complete Technical Stack Breakdown"
    ])
    
    with wf_subtab1:
        if os.path.exists("assets/project_goal_visual.jpg"):
            st.image("assets/project_goal_visual.jpg", caption="Visual Slide: Project Goal & Core Mission (Market Problem vs. ML Solution)", use_container_width=True)
            
        st.markdown("""
        #### 🎯 The Core Mission
        Buyers and sellers in the used car market face severe **information asymmetry**:
        - **Market Problem:**
          - **Blind Price Guesswork:** Lack of historical price transparency.
          - **Seller Undervaluation:** Sellers lose margin on unverified dealer trade-ins.
          - **Buyer Overpaying Risk:** Buyers cannot objectively gauge the monetary penalty of high mileage or multiple owners.
        - **Machine Learning Solution:**
          - **Data-Driven Fair Market Valuation:** Predicts fair prices in Indian Rupees (₹) with test **MAE under ₹1.5 Lakhs**.
          - **Zero Data Leakage:** Preprocessing and encodings are isolated within scikit-learn pipelines.
          - **Objective Benchmark Pricing:** Equips **Rahul (Buyer)**, **Ananya (Seller)**, and **Vikram (Dealer)** with actionable negotiation thresholds.
        """)

    with wf_subtab2:
        if os.path.exists("assets/five_stages_roadmap.jpg"):
            st.image("assets/five_stages_roadmap.jpg", caption="Visual Slide: How It Was Built - 5 Stages & 10 Sections Workflow", use_container_width=True)
            
        st.markdown("""
        #### 🏗️ The 5 Development Stages & 10 Notebook Sections
        
        1. **Stage 1: Ingestion & Cleansing (Sections 1–4)**
           - Loaded `car_data.csv` keeping the source file immutable.
           - Removed 763 duplicate records (17.6%) and filtered a recording anomaly (a 2014 car with 1 km driven).
           - Preserved high-value luxury models (e.g., Audi RS7, BMW X5) rather than clipping them.
        
        2. **Stage 2: Exploratory Data Analysis (Section 5)**
           - Plotted raw price distributions showing severe right skewness.
           - Log-transformed price (`np.log1p`) into a normalized bell curve.
           - Analyzed depreciation curves, transmission premiums, and fuel type distributions with one-line analytical findings.
        
        3. **Stage 3: Feature Engineering (Section 6)**
           - Derived `car_age` ($2020 - year$).
           - Extracted vehicle make and grouped low-frequency makes (< 25 listings) into `Other`.
           - Engineered annual driving intensity: `km_per_year` = $km\_driven / (car\_age + 1)$.
        
        4. **Stage 4: Leak-Free Preprocessing & Training (Sections 7–8)**
           - Partitioned 80% train (2,860 cars) and 20% holdout test (716 cars) with fixed seed (`random_state=42`).
           - Built `ColumnTransformer` with `StandardScaler` for numerics and `OneHotEncoder(drop='first', handle_unknown='ignore')` for categoricals.
           - Trained 4 candidate models: **Linear Regression**, **Ridge Regression**, **Random Forest**, and **Gradient Boosting**.
        
        5. **Stage 5: Evaluation & Interactive App (Sections 9–10)**
           - Inverted log predictions to Rupees using `np.expm1()`.
           - Evaluated $R^2$, MAE (₹), RMSE (₹), and 5-fold cross-validation.
           - Serialized pipelines into `models/` and launched this interactive Streamlit valuation application.
        """)

    with wf_subtab3:
        if os.path.exists("assets/tech_stack_visual.jpg"):
            st.image("assets/tech_stack_visual.jpg", caption="Visual Slide: Complete Technical Stack Breakdown (Libraries, Functions & Algorithms)", use_container_width=True)
            
        st.markdown("""
        #### 🛠️ Detailed Technology & Function Mapping
        
        | Layer | Technology | Key Functions & Classes | Specific Project Role |
        |---|---|---|---|
        | **Data Handling & Math** | `pandas`, `numpy` | `pd.read_csv`, `df.drop_duplicates()`, `df.isnull().sum()`, `df.apply()`, `np.log1p()`, `np.expm1()` | Reading data, de-duplicating 763 rows, extracting car brands, logarithmic target scaling, and inverting back to Indian Rupees. |
        | **Data Visualization** | `matplotlib`, `seaborn` | `plt.subplots()`, `sns.histplot()`, `sns.scatterplot()`, `sns.boxplot()`, `sns.heatmap()`, `sns.barplot()` | Distribution curves, price depreciation vs. age, correlation matrices, box plots by fuel/transmission, and feature importance rankings. |
        | **Machine Learning Engine** | `scikit-learn` | `StandardScaler()`, `OneHotEncoder(drop='first')`, `ColumnTransformer()`, `Pipeline()`, `train_test_split()`, `KFold()`, `cross_val_score()` | Feature scaling, dummy encoding, encapsulated leak-free pipeline architecture, 80/20 splitting, and 5-fold cross validation. |
        | **Regression Algorithms** | `scikit-learn` | `LinearRegression()`, `Ridge(alpha=1.0)`, `RandomForestRegressor()`, `GradientBoostingRegressor()` | Comparing baseline OLS, L2 regularized regression (Top model: MAE ~₹1.47L), bagged tree ensembles, and sequential gradient boosting. |
        | **Evaluation Metrics** | `scikit-learn` | `r2_score()`, `mean_absolute_error()`, `root_mean_squared_error()` | Benchmarking variance explained ($R^2$) and typical error in Indian Rupees (₹). |
        | **Serialization & Web App** | `joblib`, `streamlit` | `joblib.dump()`, `joblib.load()`, `st.set_page_config()`, `st.sidebar`, `st.metric()`, `st.tabs()`, `st.image()` | Persisting production models to disk and rendering the responsive interactive valuation UI with persona strategies and "what-if" simulations. |
        """)


