import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "customer_model_base.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "xgboost_churn_model.pkl"
SHAP_SUMMARY_PATH = BASE_DIR / "outputs" / "shap" / "shap_summary.png"
SHAP_BAR_PATH = BASE_DIR / "outputs" / "shap" / "shap_feature_importance.png"


def load_data():
    df = pd.read_csv(INPUT_PATH)

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
    return X


def generate_shap_plots():
    model = joblib.load(MODEL_PATH)
    X = load_data()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved to: {SHAP_SUMMARY_PATH}")
    print(f"SHAP bar plot saved to: {SHAP_BAR_PATH}")


if __name__ == "__main__":
    generate_shap_plots()