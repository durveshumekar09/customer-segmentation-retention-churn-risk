import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "customer_model_base.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "xgboost_churn_model.pkl"
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "churn_predictions.csv"


def load_data():
    # Load the final customer-level dataset
    df = pd.read_csv(INPUT_PATH)

    # Convert date columns if needed
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])

    return df


def prepare_features(df):
    # Use only features that do not directly reveal the churn rule
    feature_cols = [
        "frequency",
        "monetary",
        "f_score",
        "m_score",
        "total_transactions",
        "total_spend",
        "avg_order_value",
        "active_months",
        "quantity_total",
        "discount_usage_rate",
        "customer_lifespan_days",
        "clv_proxy",
    ]

    X = df[feature_cols].copy()
    y = df["churn_label"].copy()

    return X, y, feature_cols


def train_model(X, y):
    # Split the data while keeping the churn ratio similar in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Base XGBoost model
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    # Parameter search space
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # Randomized search for tuning
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Generate predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Show model performance
    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nROC-AUC Score:")
    print(round(roc_auc_score(y_test, y_pred_proba), 4))

    return best_model, X_train, X_test, y_train, y_test


def save_outputs(model, df, X, feature_cols):
    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Create churn probabilities for all customers
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    output_cols = ["customer_id", "segment", "churn_label", "churn_probability", "clv_proxy"] + feature_cols
    predictions_df = df[output_cols].copy()

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    df = load_data()
    X, y, feature_cols = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    save_outputs(model, df, X, feature_cols)