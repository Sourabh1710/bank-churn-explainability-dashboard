import streamlit as st
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings
from config import DATA_PATH, MODELS_DIR, PLOTS_DIR
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished look
st.markdown(
    """
<style>
            .main { background-color: #0e1117; }
            .metric-card {
            background: linear-gradient(135deg, #1e3a5f, #0d2137);
            border: 1px solid #2d5986;
            border-radius: 10px;
            padding: 16px 20px;
            text-align: center;
            }
            .risk-high { 
            background: linear-gradient(135deg, #5f1e1e, #3d0d0d); 
            border-color: #c0392bp; 
            }
            .risk-medium { 
            background: linear-gradient(135deg, #5f4a1e, #3d2e0d); 
            border-color: #e67e22; 
            }
            .risk-low { 
            background: linear-gradient(135deg, #1e5f2e, #0d3d1a);
             border-color: #27ae60; 
            }
            h1 {
            color: #4FC3F7 !important; 
            }
            .stButton>button {
            width: 100%; 
            background: #1565C0; 
            color: white; 
            border: none;
            border-radius: 8px; 
            padding:12px; 
            font-size: 16px; 
            font-weight: bold; 
            }
            .stButton>button:hover {
            background: #1976D2; 
            }
</style>
""",
    unsafe_allow_html=True
)


# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    models_dir = "models"
    return {
        "model": joblib.load(f"{models_dir}/xgb_model.pkl"),
        "rf_model": joblib.load(f"{models_dir}/rf_model.pkl"),
        "preprocessor": joblib.load(f"{models_dir}/preprocessor.pkl"),
        "feature_names": joblib.load(f"{models_dir}/feature_names.pkl"),
        "explainer": joblib.load(f"{models_dir}/shap_explainer.pkl"),
        "summary": joblib.load(f"{models_dir}/model_summary.pkl"),
    }


# Feature Engineering
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["credit_score_bucket"] = pd.cut(
        df["CreditScore"],
        bins=[0, 579, 669, 739, 799, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    )
    df["tenure_age_ratio"] = df["Tenure"] / (df["Age"] + 1e-9)
    df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1e-9)
    df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    return df


# Helper : risk label & color
def risk_label(prob):
    if prob >= 0.65:
        return "High Risk", "risk-high", "#e74c3c"
    elif prob >= 0.35:
        return "Medium Risk", "risk-medium", "#e67e22"
    else:
        return "Low Risk", "risk-low", "#27ae60"


## APP Layout

# Header
st.title("🏦 Bank Customer Churn Predictor")
st.markdown(
    "**XGBoost + SHAP Explainability** - Predict Which customers are at risk of leaving"
)
st.markdown("---")

# Load Artifacts
try:
    arts = load_artifacts()
    model = arts["model"]
    rf_model = arts["rf_model"]
    preprocessor = arts["preprocessor"]
    feature_names = arts["feature_names"]
    explainer = arts["explainer"]
    summary = arts["summary"]
except FileNotFoundError:
    st.error(" Model Artifacts not found.")
    st.stop()

# Sidebar : Model Performance Dashboard
with st.sidebar:
    st.header("Model Performance")
    st.markdown(f"""
    | Model | AUC-ROC |
    |---|---|
    | XGBoost (tuned) | **{summary['xgb_auc']:.4f}** |
    | Random Forest (tuned) | {summary['rf_auc']:.4f} |
    | Logistic Regression | {summary['lr_auc']:.4f} |
     """)
    st.caption(
        f"Trained on {summary['train_size']:,} records - Evaluated on {summary['test_size']:,}"
    )
    st.markdown("---")
    st.header("Dataset Stats")
    st.metric("Total Records", "10,000")
    st.metric("Features Used", summary["num_features"])
    st.metric("Churn Rate", f"{summary['churn_rate']:.1%}")
    st.markdown("---")
    st.header("Model Selection")
    model_choice = st.radio("Predict with:", ["XGBoost (Best)", "Random Forest"])
    st.markdown("---")
    st.caption("Built with scikit-learn, XGBoost, SHAP, Optuna, Streamlit")

tab1, tab2, tab3 = st.tabs(["Predict Customer", " EDA Visuals", " SHAP Deep Dive"])

# TAB 1 : Predict Customer
with tab1:
    st.subheader("Enter Customer Details")
    st.markdown(
        "Adjust the sliders to match a customer profile, then click **Predict**."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("** Personal Info**")
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 92, 40)
        credit_score = st.slider("Credit Score", 300, 850, 650)
    with col2:
        st.markdown("** Financial Profile**")
        balance = st.number_input("Account Balance (₹)", 0, 250_000, 80_000, step=1_000)
        salary = st.number_input(
            "Estimated Salary (₹)", 0, 200_000, 100_000, step=1_000
        )
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_credit_card = st.checkbox("Has Credit Card", value=True)
    with col3:
        st.markdown("**Banking Behavious**")
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        is_active = st.checkbox("Active Member", value=True)

    st.markdown("---")

    if st.button(" Predict Churn Risk"):
        raw_input = pd.DataFrame(
            [
                {
                    "CreditScore": credit_score,
                    "Geography": geography,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": float(balance),
                    "NumOfProducts": num_products,
                    "HasCrCard": int(has_credit_card),
                    "IsActiveMember": int(is_active),
                    "EstimatedSalary": float(salary),
                }
            ]
        )

        # Applying Feature Engineering
        raw_fe = engineer_features(raw_input)

        # Preprocess
        X_input = preprocessor.transform(raw_fe)

        # Predict
        active_model = model if model_choice == "XGBoost (Best)" else rf_model
        proba = float(active_model.predict_proba(X_input)[0, 1])
        label, css_class, color = risk_label(proba)

        # Result Display
        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            st.markdown(
                f"""
            <div class="metric-card {css_class}">
                <h2 style="color:{color}; 
                margin:0">{label}
                </h2>
                <p style="font-size:36px; font-weight:bold; color:{color}; margin:8px 0">
                    {proba:.1%}
                </p>
                <p style="color:#aaa; margin:0">Churn Probability</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with r2:
            fig_g, ax_g = plt.subplots(figsize=(3, 2.5))
            fig_g.patch.set_facecolor("0e1117")
            ax_g.barh(["Churn Risk"], [proba], color=color, height=0.4)
            ax_g.barh(
                ["Churn Risk"], [1 - proba], left=proba, color="#2d2d2d", height=0.4
            )
            ax_g.set_xlim(0, 1)
            ax_g.set_facecolor("#0e1117")
            ax_g.tick_params(colors="white")
            ax_g.xaxis.label.set_color("white")
            ax_g.set_xlabel("Probability", color="white")
            ax_g.spines[:].set_visible(False)
            st.pyplot(fig_g, use_container_width=True)
            plt.close()
        with r3:
            # Business recommendation
            if proba >= 0.65:
                recommendation = """
                ** Immediate Action Required**
                - Schedule personal advisor call
                - Offer premium product bundle
                - Apply retention discount
                - Fast-track complaint resolution
                """
            elif proba >= 0.35:
                recommendation = """
                ** Monitor & Engage**
                - Send personalised product offer
                - Check recent support tickets
                - Trigger loyalty reward email
                - Cross-sell savings product
                """
            else:
                recommendation = """
                ** Customer Looks Healthy**
                - Continue standard engagement
                - Opportunity for upselling
                - Low priority for retention spend
                """
            st.info(recommendation)

        st.markdown("### Why did the model give this score?")
        st.caption(
            "SHAP values show which features pushed the churn probability up (red) or down (blue)"
        )

        if model_choice == "XGBoost (Best)":
            shap_vals_customer = explainer.shap_values(X_input)[0]
            expected_value = explainer.expected_value

            shap_df = (
                pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "SHAP Value": shap_vals_customer,
                        "Feature Val": X_input[0],
                    }
                )
                .sort_values("SHAP Value", key=abs, ascending=False)
                .head(12)
            )

            fig_w, ax_w = plt.subplots(figsize=(9, 5))
            fig_w.patch.set_facecolor("#0e1117")
            ax_w.set_facecolor("#0e1117")

            bar_colors = [
                "#e74c3c" if v > 0 else "#3498db" for v in shap_df["SHAP Value"]
            ]
            bars = ax_w.barh(
                shap_df["Feature"],
                shap_df["SHAP Value"],
                color=bar_colors,
                edgecolor="white",
                linewidth=0.3,
            )
            ax_w.axvline(x=0, color="white", linewidth=1, alpha=0.5)
            ax_w.set_xlabel("SHAP Value (impact on churn probability)", color="white")
            ax_w.set_title(
                f"Feature Contributions for This Customer\nBase rate: {expected_value:.2f} - Prediction: {proba:.2f}",
                color="white",
                fontsize=11,
            )
            ax_w.tick_params(colors="white")
            ax_w.spines[:].set_color("#444")
            plt.tight_layout()
            st.pyplot(fig_w, use_container_width=True)
            plt.close()
        else:
            st.info(
                "SHAP waterfall available for XGBoost model. Switch model in sidebar."
            )

        # Feature Summary Table
        st.markdown("### Engineered Features for This Customer")
        eng_summary = pd.DataFrame(
            {
                "Feature": [
                    "Credit Score Bucket",
                    "Tenure/Age Ratio",
                    "Balance/Salary Ratio",
                    "Products/Tenure",
                ],
                "Value": [
                    str(raw_fe["credit_score_bucket"].iloc[0]),
                    f"{raw_fe['tenure_age_ratio'].iloc[0]:.3f}",
                    f"{raw_fe['balance_salary_ratio'].iloc[0]:.3f}",
                    f"{raw_fe['products_per_tenure'].iloc[0]:.3f}",
                ],
                "Meaning": [
                    "Creditworthiness band (Poor -> Excellent)",
                    "Higher = loyal member relatives to age",
                    "Higher = deeper financial relationship",
                    "Higher = cross-sold quickly",
                ],
            }
        )
        st.dataframe(eng_summary, use_container_width=True, hide_index=True)

# TAB 2: EDA Visuals
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.markdown("Pre-computed visualisations from the training pipeline.")

    plots = [
        ("outputs/01_eda_overview.png", "EDA Overview - Churn by Key Segments"),
        ("outputs/02_correlation_heatmap.png", "Feature Correlation Heatmap"),
        ("outputs/03_roc_pr_curves.png", "ROC & Precision-Recall Curves"),
        ("outputs/04_confusion_matrices.png", "Confusion Matrices"),
        (
            "outputs/07_business_insights.png",
            "Business Insights - Feature Importance",
        ),
    ]

    for path, caption in plots:
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"Plot not found: {path}.")

# TAb 3 : SHAP Deep Dive
with tab3:
    st.subheader("SHAP Global Explainability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows **globally** which features most
    influence churn predictions across the entire test set.

    - **Top plot**: Mean |SHAP| — overall importance
    - **Summary plot**: Direction of impact (red = increases churn probability)
    """)

    shap_plots = [
        ("outputs/05_shap_importance.png", "SHAP Feature Importance (Bar)"),
        (
            "outputs/06_shap_summary.png",
            "SHAP Summary (Beeswarm) - Direction & Magnitude",
        ),
    ]

    for path, caption in shap_plots:
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"Plot not found: {path}.")
    st.info("""
    **How to read the SHAP Summary Plot:**
    - Each dot = one customer in the test set
    - **X-axis**: SHAP value (how much that feature pushed the prediction)
    - **Colour**: Feature value (red = high, blue = low)
    - Example: `Age` — red dots (older customers) on the right → higher churn
    """)
