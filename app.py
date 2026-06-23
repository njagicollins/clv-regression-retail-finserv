import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CLV Predictor", layout="wide")
st.title("Customer Lifetime Value Prediction")
st.caption("Interactive CLV prediction using your trained XGBoost model")

# -----------------------------
# Load model + feature names
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("clv_model.pkl")

model = load_model()
st.success("Model loaded!")

# def load_model():
#     model = joblib.load("clv_model.pkl")
#     #features = joblib.load("feature_names.pkl")
#     return model #features

# model, _ = load_model()
# st.success("Model loaded!")

#model, expected_features = load_model()
#st.success(f"Model loaded! Expects {len(expected_features)} features")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Customer Information")

gender = st.selectbox("Gender", ["Male", "Female"])
customer_segment = st.selectbox("Customer Segment", ["Basic", "Standard", "Premium"])
recency_bucket = st.selectbox("Recency Bucket", ["Active", "At_Risk", "Cold", "Very_Active"])

tenure_months = st.number_input("Tenure (months)", min_value=0.0, step=1.0)
avg_monthly_spend = st.number_input("Avg Monthly Spend ($)", min_value=0.0)
purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0)
num_products_owned = st.number_input("Number of Products Owned", min_value=0)
engagement_score = st.number_input("Engagement Score", min_value=0.0)
churn_risk_score = st.number_input("Churn Risk Score", min_value=0.0)
email_open_rate = st.number_input("Email Open Rate (%)", min_value=0.0)

# -----------------------------
# Engineered Features
# -----------------------------
spend_per_product = (
    avg_monthly_spend / num_products_owned if num_products_owned > 0 else 0
)

engagement_x_spend = engagement_score * avg_monthly_spend

# -----------------------------
# Missingness Flags (default = 0)
# -----------------------------
discount_usage_rate_missing = 0
days_since_last_purchase_missing = 0
churn_risk_missing = 0
email_open_rate_missing = 0

# -----------------------------
# Build DataFrame for prediction
# -----------------------------
input_data = pd.DataFrame([{
    'gender': gender,
    'customer_segment': customer_segment,
    'tenure_months': tenure_months,
    'avg_monthly_spend': avg_monthly_spend,
    'purchase_frequency': purchase_frequency,
    'num_products_owned': num_products_owned,
    'engagement_score': engagement_score,
    'churn_risk_score': churn_risk_score,
    'email_open_rate': email_open_rate,

    'spend_per_product': spend_per_product,
    'engagement_x_spend': engagement_x_spend,

    'recency_bucket': recency_bucket,

    'discount_usage_rate_missing': discount_usage_rate_missing,
    'days_since_last_purchase_missing': days_since_last_purchase_missing,
    'churn_risk_missing': churn_risk_missing,
    'email_open_rate_missing': email_open_rate_missing
}])

# Reorder columns to match model
#input_data = input_data.reindex(columns=expected_features)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict CLV"):
    preds_log = model.predict(input_data)
    preds = np.expm1(preds_log)

    st.subheader("Predicted Customer Lifetime Value")
    st.metric(label="CLV ($)", value=f"{preds[0]:,.2f}")
