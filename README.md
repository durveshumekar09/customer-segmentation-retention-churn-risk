# Customer Segmentation, Retention and Churn Risk Dashboard

## Project Overview

This project is an end-to-end customer analytics workflow built on a synthetic e-commerce transaction dataset. It combines SQL, SQLite, Python, machine learning, explainability, and dashboarding to analyze customer behaviour, retention trends, churn risk, and customer value.

The workflow was designed to cover four main analytical areas:

- customer-level RFM feature engineering
- cohort retention analysis
- churn prediction using XGBoost
- interactive dashboarding with Streamlit

---

## Objectives

The project aims to:

- transform transaction-level data into customer-level analytical features
- segment customers into meaningful behavioural groups
- measure cohort retention over a 12-month period
- build a churn prediction model and explain its drivers
- compare churn risk and CLV proxy across customer segments
- present insights in an interactive dashboard

---

## Dataset

- **Type:** Synthetic e-commerce transaction dataset
- **Transactions:** 14,500
- **Customers:** 1,200
- **Date Range:** 2025-01-05 to 2025-12-31

### Raw fields used

- `transaction_id`
- `customer_id`
- `transaction_date`
- `order_value`
- `quantity`
- `product_category`
- `payment_method`
- `channel`
- `discount_used_pct`
- `customer_profile_segment`
- `customer_city`
- `customer_state`
- `signup_date`

---

## Tools and Technologies

- SQL
- SQLite
- Python
- pandas
- scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly
- joblib

---

## Project Workflow

### 1. Data Loading and SQLite Setup
The raw CSV file is loaded into a SQLite database as a `transactions` table. This supports SQL-based analysis, validation, and reproducibility.

### 2. RFM Feature Engineering
Customer-level RFM metrics were created:

- **Recency:** days since last purchase
- **Frequency:** total transaction count
- **Monetary:** total customer spend

RFM outputs were generated and checked using both SQL and Python.

### 3. Behavioural Segmentation
Customers were grouped into four RFM-based behavioural segments:

- High-Value
- Repeat
- Low-Engagement
- At-Risk

These segments were created using rule-based RFM scoring so the segmentation logic remains explainable in business terms.

### 4. Cohort Retention Analysis
Monthly cohort retention logic was built using first purchase month and repeat monthly activity. A cohort retention table was created to track 12-month retention behaviour across acquisition cohorts.

### 5. Churn Label and CLV Proxy Creation
A churn label was defined using a 90-day inactivity rule:

- `churn_label = 1` if the customer made no purchase in the last 90 days
- `churn_label = 0` otherwise

A CLV proxy was also created using transaction frequency, average order value, and active months.

### 6. Churn Prediction Model
An XGBoost binary classifier was trained to predict churn risk using customer-level behavioural and value-related features. Model tuning was performed using `RandomizedSearchCV`.

### 7. Model Explainability
SHAP was used to interpret model behaviour and identify the features with the strongest influence on churn predictions.

### 8. Interactive Dashboard
A Streamlit dashboard was built to visualize:

- segment performance
- 12-month cohort retention
- churn-risk distribution
- CLV proxy comparisons
- customer-level filtered views
- SHAP model interpretability outputs

---

## Verified Dataset and Validation Summary

SQL validation queries confirmed:

- **Total transactions:** 14,500
- **Total customers:** 1,200
- **Date range:** 2025-01-05 to 2025-12-31
- **Nulls in key fields:** 0
- **Duplicate transaction IDs:** none found
- **Non-positive order values:** 0
- **Non-positive quantities:** 0
- **One-time customers:** 3
- **Repeat customers:** 1,197

### Distribution checks

**Channel distribution**
- Web: 5,776
- App: 5,576
- Store: 3,148

**Top product category distribution**
- Fashion: 3,167
- Groceries: 2,915
- Electronics: 2,589
- Home: 2,312
- Beauty: 1,766
- Sports: 1,751

---

## Segmentation Output

The final RFM-based segmentation split 1,200 customers into:

- **High-Value:** 391
- **At-Risk:** 385
- **Low-Engagement:** 275
- **Repeat:** 149

This segmentation supports comparison of customer value, frequency, and churn patterns across groups.

---

## Churn Label Distribution

Using the 90-day inactivity rule, the final customer base was labeled as:

- **Active customers (0):** 1,083
- **Churned customers (1):** 117

---

## Model Performance

After removing leakage-prone features from training, the final churn model achieved:

- **ROC-AUC:** 0.9473
- **Accuracy:** 0.91
- **Precision for churned customers:** 0.51
- **Recall for churned customers:** 0.78
- **F1-score for churned customers:** 0.62

### Best Parameters

- `subsample = 0.8`
- `n_estimators = 200`
- `max_depth = 4`
- `learning_rate = 0.05`
- `colsample_bytree = 0.8`

### Confusion Matrix

- **True Negatives:** 200
- **False Positives:** 17
- **False Negatives:** 5
- **True Positives:** 18

These results show that the model performs well in identifying churn-prone customers while keeping evaluation realistic.

---

## SHAP-Based Model Interpretation

SHAP analysis showed that the most influential features in the churn model were:

- `customer_lifespan_days`
- `avg_order_value`
- `monetary`
- `clv_proxy`

The SHAP summary plots indicate that shorter customer lifespan and lower value-related metrics were associated with higher predicted churn risk. This aligns with expected customer behaviour patterns in retention analysis.

---

## SQL Components

The project includes SQL-based analysis files for:

- `rfm_queries.sql`
- `cohort_queries.sql`
- `churn_label_queries.sql`
- `validation_queries.sql`

These were executed against the SQLite database and exported as processed outputs, making the SQL layer reproducible and part of the actual workflow rather than only documentation.

---

## Folder Structure

```text
customer-segmentation-dashboard/
│
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── customer_transactions_synthetic_2025.csv
│   └── processed/
├── database/
│   └── customer_project.db
├── outputs/
│   ├── models/
│   │   └── xgboost_churn_model.pkl
│   └── shap/
│       ├── shap_feature_importance.png
│       └── shap_summary.png
├── sql/
│   ├── rfm_queries.sql
│   ├── cohort_queries.sql
│   ├── churn_label_queries.sql
│   └── validation_queries.sql
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   ├── customer_metrics.py
│   ├── churn_model.py
│   ├── shap_analysis.py
│   ├── sql_runner.py
│   └── validation_runner.py
├── requirements.txt
├── README.md
└── .gitignore