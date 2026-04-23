import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# Page setup
st.set_page_config(
    page_title="Customer Segmentation, Retention and Churn Risk Dashboard",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SEGMENTS_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "segment_summary.csv"
COHORT_PATH = BASE_DIR / "data" / "processed" / "cohort_retention.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "churn_predictions.csv"
SHAP_IMPORTANCE_PATH = BASE_DIR / "outputs" / "shap" / "shap_feature_importance.png"
SHAP_SUMMARY_PATH = BASE_DIR / "outputs" / "shap" / "shap_summary.png"


@st.cache_data
def load_data():
    segments_df = pd.read_csv(SEGMENTS_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)
    cohort_df = pd.read_csv(COHORT_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    return segments_df, summary_df, cohort_df, predictions_df


def format_number(value):
    return f"{value:,.0f}"


def format_decimal(value):
    return f"{value:.2f}"


segments_df, summary_df, cohort_df, predictions_df = load_data()

# Dashboard title
st.title("Customer Segmentation, Retention and Churn Risk Dashboard")
st.write(
    "Interactive dashboard for behavioural segmentation, cohort retention, churn prediction, "
    "and customer value analysis."
)

# Sidebar filters
st.sidebar.header("Filters")

all_segments = sorted(predictions_df["segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Segment",
    options=all_segments,
    default=all_segments
)

churn_range = st.sidebar.slider(
    "Churn probability range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05
)

clv_min = float(predictions_df["clv_proxy"].min())
clv_max = float(predictions_df["clv_proxy"].max())
selected_clv = st.sidebar.slider(
    "CLV proxy range",
    min_value=float(round(clv_min, 0)),
    max_value=float(round(clv_max, 0)),
    value=(float(round(clv_min, 0)), float(round(clv_max, 0))),
    step=100.0
)

top_n = st.sidebar.slider(
    "Rows to show in customer tables",
    min_value=10,
    max_value=100,
    value=25,
    step=5
)

selected_metric = st.sidebar.selectbox(
    "Metric for segment comparison",
    options=["customers", "avg_churn_probability", "avg_clv_proxy", "avg_frequency", "avg_monetary"],
    index=0
)

filtered_predictions = predictions_df[
    (predictions_df["segment"].isin(selected_segments)) &
    (predictions_df["churn_probability"] >= churn_range[0]) &
    (predictions_df["churn_probability"] <= churn_range[1]) &
    (predictions_df["clv_proxy"] >= selected_clv[0]) &
    (predictions_df["clv_proxy"] <= selected_clv[1])
].copy()

if filtered_predictions.empty:
    st.warning("No records match the selected filters.")
    st.stop()

filtered_predictions["risk_band"] = pd.cut(
    filtered_predictions["churn_probability"],
    bins=[0, 0.3, 0.7, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

# KPI row
total_customers = filtered_predictions["customer_id"].nunique()
avg_churn_probability = filtered_predictions["churn_probability"].mean()
avg_clv_proxy = filtered_predictions["clv_proxy"].mean()
high_risk_customers = filtered_predictions[
    filtered_predictions["churn_probability"] >= 0.7
]["customer_id"].nunique()
avg_frequency = filtered_predictions["frequency"].mean()
avg_monetary = filtered_predictions["monetary"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Customers", format_number(total_customers))
k2.metric("Average Churn Probability", format_decimal(avg_churn_probability))
k3.metric("Average CLV Proxy", format_number(avg_clv_proxy))
k4.metric("High-Risk Customers", format_number(high_risk_customers))
k5.metric("Average Frequency", format_decimal(avg_frequency))
k6.metric("Average Monetary", format_number(avg_monetary))

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Segment Performance",
    "Cohort Retention",
    "Churn Analysis",
    "CLV Analysis",
    "Model Interpretability",
    "Customer Table"
])

# Tab 1
with tab1:
    st.subheader("Segment Performance")

    segment_perf = (
        filtered_predictions.groupby("segment")
        .agg(
            customers=("customer_id", "nunique"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_clv_proxy=("clv_proxy", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values(by="customers", ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_segment_count = px.bar(
            segment_perf,
            x="segment",
            y="customers",
            color="segment",
            title="Customers by Segment",
            text_auto=True
        )
        fig_segment_count.update_layout(showlegend=False)
        st.plotly_chart(fig_segment_count, use_container_width=True)

    with c2:
        fig_selected_metric = px.bar(
            segment_perf,
            x="segment",
            y=selected_metric,
            color="segment",
            title=f"Segment Comparison: {selected_metric}"
        )
        fig_selected_metric.update_layout(showlegend=False)
        st.plotly_chart(fig_selected_metric, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig_segment_scatter = px.scatter(
            segment_perf,
            x="avg_frequency",
            y="avg_monetary",
            size="customers",
            color="segment",
            title="Average Frequency vs Average Monetary"
        )
        st.plotly_chart(fig_segment_scatter, use_container_width=True)

    with c4:
        fig_segment_churn = px.bar(
            segment_perf,
            x="segment",
            y="avg_churn_probability",
            color="segment",
            title="Average Churn Probability by Segment",
            text_auto=".2f"
        )
        fig_segment_churn.update_layout(showlegend=False)
        st.plotly_chart(fig_segment_churn, use_container_width=True)

    st.dataframe(segment_perf, use_container_width=True)

# Tab 2
with tab2:
    st.subheader("12-Month Cohort Retention")

    cohort_filtered = cohort_df[cohort_df["cohort_index"] <= 12].copy()

    available_cohorts = sorted(cohort_filtered["cohort_month"].unique().tolist())
    selected_cohorts = st.multiselect(
        "Cohort month",
        options=available_cohorts,
        default=available_cohorts
    )

    cohort_filtered = cohort_filtered[
        cohort_filtered["cohort_month"].isin(selected_cohorts)
    ]

    c1, c2 = st.columns([2, 1])

    with c1:
        fig_retention_curve = px.line(
            cohort_filtered,
            x="cohort_index",
            y="retention_rate",
            color="cohort_month",
            markers=True,
            title="Cohort Retention Curves"
        )
        fig_retention_curve.update_layout(
            xaxis_title="Cohort Month Number",
            yaxis_title="Retention Rate"
        )
        st.plotly_chart(fig_retention_curve, use_container_width=True)

    with c2:
        latest_retention = (
            cohort_filtered.sort_values(["cohort_month", "cohort_index"])
            .groupby("cohort_month")
            .tail(1)
            .sort_values("retention_rate", ascending=False)
        )

        fig_latest_retention = px.bar(
            latest_retention,
            x="cohort_month",
            y="retention_rate",
            title="Latest Available Retention by Cohort",
            text_auto=".2f"
        )
        st.plotly_chart(fig_latest_retention, use_container_width=True)

    cohort_pivot = cohort_filtered.pivot(
        index="cohort_month",
        columns="cohort_index",
        values="retention_rate"
    )

    fig_retention_heatmap = px.imshow(
        cohort_pivot,
        aspect="auto",
        title="Cohort Retention Heatmap",
        labels={"x": "Cohort Month Number", "y": "Cohort Month", "color": "Retention Rate"}
    )
    st.plotly_chart(fig_retention_heatmap, use_container_width=True)

    st.dataframe(cohort_filtered, use_container_width=True)

# Tab 3
with tab3:
    st.subheader("Churn Analysis")

    c1, c2 = st.columns(2)

    with c1:
        fig_churn_dist = px.histogram(
            filtered_predictions,
            x="churn_probability",
            nbins=30,
            color="risk_band",
            title="Churn Probability Distribution"
        )
        st.plotly_chart(fig_churn_dist, use_container_width=True)

    with c2:
        fig_churn_box = px.box(
            filtered_predictions,
            x="segment",
            y="churn_probability",
            color="segment",
            title="Churn Probability by Segment"
        )
        fig_churn_box.update_layout(showlegend=False)
        st.plotly_chart(fig_churn_box, use_container_width=True)

    risk_summary = (
        filtered_predictions.groupby("risk_band", observed=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_clv_proxy=("clv_proxy", "mean")
        )
        .reset_index()
    )

    c3, c4 = st.columns(2)

    with c3:
        fig_risk_band = px.bar(
            risk_summary,
            x="risk_band",
            y="customers",
            color="risk_band",
            title="Customers by Risk Band",
            text_auto=True
        )
        fig_risk_band.update_layout(showlegend=False)
        st.plotly_chart(fig_risk_band, use_container_width=True)

    with c4:
        high_risk_table = filtered_predictions.sort_values(
            by="churn_probability",
            ascending=False
        ).head(top_n)

        st.write("Top high-risk customers")
        st.dataframe(
            high_risk_table[
                ["customer_id", "segment", "churn_probability", "clv_proxy", "frequency", "monetary"]
            ],
            use_container_width=True
        )

    st.dataframe(risk_summary, use_container_width=True)

# Tab 4
with tab4:
    st.subheader("CLV Proxy Analysis")

    c1, c2 = st.columns(2)

    with c1:
        fig_clv_scatter = px.scatter(
            filtered_predictions,
            x="clv_proxy",
            y="churn_probability",
            color="segment",
            hover_data=["customer_id"],
            title="CLV Proxy vs Churn Probability"
        )
        st.plotly_chart(fig_clv_scatter, use_container_width=True)

    with c2:
        fig_clv_box = px.box(
            filtered_predictions,
            x="segment",
            y="clv_proxy",
            color="segment",
            title="CLV Proxy by Segment"
        )
        fig_clv_box.update_layout(showlegend=False)
        st.plotly_chart(fig_clv_box, use_container_width=True)

    top_clv = filtered_predictions.sort_values(
        by="clv_proxy",
        ascending=False
    ).head(top_n)

    bottom_clv = filtered_predictions.sort_values(
        by="clv_proxy",
        ascending=True
    ).head(top_n)

    c3, c4 = st.columns(2)

    with c3:
        st.write("Top customers by CLV proxy")
        st.dataframe(
            top_clv[
                ["customer_id", "segment", "clv_proxy", "churn_probability", "frequency", "monetary"]
            ],
            use_container_width=True
        )

    with c4:
        st.write("Lowest customers by CLV proxy")
        st.dataframe(
            bottom_clv[
                ["customer_id", "segment", "clv_proxy", "churn_probability", "frequency", "monetary"]
            ],
            use_container_width=True
        )

# Tab 5
with tab5:
    st.subheader("Model Interpretability")

    c1, c2 = st.columns(2)

    with c1:
        st.write("SHAP Feature Importance")
        if SHAP_IMPORTANCE_PATH.exists():
            st.image(str(SHAP_IMPORTANCE_PATH), use_container_width=True)
        else:
            st.info("SHAP feature importance image not found.")

    with c2:
        st.write("SHAP Summary Plot")
        if SHAP_SUMMARY_PATH.exists():
            st.image(str(SHAP_SUMMARY_PATH), use_container_width=True)
        else:
            st.info("SHAP summary image not found.")

    st.write(
        "These plots help explain which features contribute most to churn predictions "
        "and whether model behaviour aligns with customer purchase patterns."
    )

# Tab 6
with tab6:
    st.subheader("Customer-Level Table")

    sort_by = st.selectbox(
        "Sort table by",
        options=["churn_probability", "clv_proxy", "frequency", "monetary"],
        index=0
    )

    ascending = st.radio(
        "Sort order",
        options=["Descending", "Ascending"],
        index=0,
        horizontal=True
    )

    table_df = filtered_predictions.sort_values(
        by=sort_by,
        ascending=True if ascending == "Ascending" else False
    ).head(top_n)

    st.dataframe(table_df, use_container_width=True)

    csv_data = filtered_predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_customer_view.csv",
        mime="text/csv"
    )