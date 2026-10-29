# Used Cars - Regression Analysis: Project Idea

## 1. One-Line Idea
Build machine learning models to predict used car prices using feature engineering, data preprocessing, and regression techniques, developed in Antigravity as one clear, section-by-section notebook.

## 2. Problem Statement
Buyers and sellers of used cars often do not know what a fair price is. Price depends on many factors at once: age, kilometers driven, brand, fuel type, transmission, and number of previous owners. This project builds regression models that estimate a car's selling price from these features, compares the models, and explains which features matter most.

## 3. Goals
- Clean and prepare real used-car listing data
- Create useful features such as car age and brand
- Train and compare four regression models: Linear Regression, Ridge, Random Forest, and Gradient Boosting
- Evaluate models fairly with R2, MAE, and RMSE on unseen test data
- Identify the features that influence price the most
- Finish with a clear conclusion that a non-technical person can understand

## 4. Key Questions to Answer
1. How do car age and kilometers driven affect the selling price?
2. Do fuel type, transmission, and ownership history change the price?
3. Which brands hold their value best?
4. Which regression model predicts price most accurately?
5. How large is the typical prediction error (MAE) in rupees?
6. Which features are the most important for prediction?

## 5. Dataset
- **Source:** Vehicle dataset from CarDekho (Kaggle, uploaded by nehalbirla). It contains several CSV files, so choose the most detailed one after checking the columns.
- **Typical columns:** name, year, selling_price, km_driven, fuel, seller_type, transmission, owner. Some files also include mileage, engine, max power, and seats.
- **Target variable:** selling_price
- **Rule:** keep the original CSV unchanged. All fixes happen in the notebook.
- **First check:** run `df.shape` and `df.columns` to confirm which version you downloaded, because row and column counts differ between files.

## 6. Project Setup in Antigravity

### Keep it simple: one folder, one notebook
```
used-cars/
├── car_data.csv               <- the dataset
├── used_car_price.ipynb       <- all code, top to bottom
├── AGENTS.md                  <- rules for the AI agent (optional but useful)
└── idea.md                    <- this file
```
Add `models/`, `app/`, and `README.md` only after the notebook is finished, and only if you build a Streamlit app.

### Setup steps
1. Open Antigravity and open the `used-cars` folder as your workspace.
2. Open the terminal and create an environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac/Linux
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Copy the CSV into the folder.
4. Create `used_car_price.ipynb` and select the `venv` Python kernel.

### Libraries (kept to the minimum)
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Suggested AGENTS.md
```markdown
# Agent Rules
- Project: used car price prediction (regression) in ONE notebook: used_car_price.ipynb
- Use only pandas, numpy, matplotlib, seaborn, scikit-learn
- Do not create extra folders or files unless asked
- Keep cells short, one purpose per cell, with a comment explaining why
- Never modify the original CSV
- Fit encoders and scalers on training data only (avoid data leakage)
- Add a markdown heading before each section
- After each section, summarize what was done in 2-3 lines
```

## 7. Notebook Workflow (10 Sections)

| # | Section | What it contains | Output |
|---|---|---|---|
| 1 | Imports | All libraries in one cell | Notebook runs without import errors |
| 2 | Load data | CSV path and `pd.read_csv` | DataFrame `df` |
| 3 | Inspect | `head()`, `info()`, `describe()`, nulls, duplicates | List of data problems |
| 4 | Clean | Fix missing values, wrong types, duplicates, impossible values | Clean DataFrame |
| 5 | EDA | Price distribution, price vs. age, price vs. km, price by fuel, transmission, and owner | Charts with one-line findings |
| 6 | Feature engineering | Car age, brand from name, optional log of price, km per year | New columns |
| 7 | Preprocessing | Encoding categories, train/test split, scaling where needed | `X_train`, `X_test`, `y_train`, `y_test` |
| 8 | Models | Linear Regression, Ridge, Random Forest, Gradient Boosting | Four trained models |
| 9 | Evaluation | R2, MAE, RMSE table, actual vs. predicted plot, feature importance | Model comparison table |
| 10 | Conclusion | Best model, key drivers of price, limitations, next steps | Written summary |

### Section details

**1. Imports**
Put every import in one cell: pandas, numpy, matplotlib, seaborn, and the scikit-learn tools (`train_test_split`, models, metrics, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`).

**2. Load data**
```python
df = pd.read_csv("car_data.csv")
```

**3. Inspect**
- `df.shape`, `df.head()`, `df.info()`, `df.describe()`
- `df.isnull().sum()` and `df.duplicated().sum()`
- Write down every problem you find.

**4. Clean**
- Drop duplicate rows
- Fill or drop missing values and document the choice
- Convert text columns that hold numbers (for example "1248 CC" or "19.4 kmpl") into numeric values, if your file has them
- Check for impossible values (zero or negative price, unrealistic km driven)
- Review extreme outliers in price and decide whether to cap or remove them

**5. EDA**
- Histogram of selling price (usually skewed right)
- Scatter: price vs. year and price vs. km driven
- Box plots: price by fuel, transmission, seller type, and owner
- Correlation heatmap of numeric columns
- Write one sentence under every chart

**6. Feature engineering**
- `car_age = current_year - year`
- `brand` = first word of the car name (group rare brands as "Other")
- `log_price = log1p(selling_price)` to reduce skew (convert predictions back with `expm1`)
- Optional: km per year

**7. Preprocessing**
- Choose features `X` and target `y`
- Split into train and test sets (80/20, `random_state=42`)
- One-hot encode categorical columns and scale numeric columns for the linear models
- Use a `Pipeline` with `ColumnTransformer` so that encoders are fit on training data only

**8. Models**
Train in this order, from simplest to most complex:
1. Linear Regression (baseline)
2. Ridge Regression (regularized)
3. Random Forest Regressor
4. Gradient Boosting Regressor

**9. Evaluation**
- Calculate R2, MAE, and RMSE for every model on the test set
- Compare train score vs. test score to detect overfitting
- Show a results table sorted by R2
- Plot actual vs. predicted prices for the best model
- Plot feature importance (Random Forest or Gradient Boosting)
- Optional: 5-fold cross-validation for the top two models

**10. Conclusion**
Answer each key question from section 4 in plain language, name the best model and its typical error, state the limits (small dataset, listings are not final sale prices), and suggest next steps.
