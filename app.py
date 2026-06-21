import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load model and metadata
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_credit_score_model.pkl"
META_PATH  = ARTIFACTS_DIR / "model_meta.json"

@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return pipeline, meta


pipeline, meta = load_model()

LABEL_MAP      = {0: "Poor", 1: "Standard", 2: "Good"}
LABEL_COLORS   = {"Poor": "#FF4B4B", "Standard": "#FFA500", "Good": "#21BA45"}
BEST_MODEL_NAME = meta.get("best_model_name", "Best Model")
TRAIN_COLUMNS  = meta.get("train_columns", [])

LOAN_TYPES = ["Student Loan", "Mortgage Loan", "Debt Consolidation Loan",
              "Payday Loan", "Credit-Builder Loan", "Personal Loan",
              "Home Equity Loan", "Auto Loan", "Not Specified",]

OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
               "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
               "Accountant", "Musician", "Mechanic", "Writer", "Architect",]

PAYMENT_BEHAVIOURS = [
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
]

# Inference helper
def build_input_df(inputs: dict) -> pd.DataFrame:
    row = {
        "Age": inputs["age"],
        "Annual_Income": inputs["annual_income"],
        "Monthly_Inhand_Salary": inputs["monthly_salary"],
        "Num_Bank_Accounts": inputs["num_bank_accounts"],
        "Num_Credit_Card": inputs["num_credit_cards"],
        "Interest_Rate": inputs["interest_rate"],
        "Num_of_Loan": inputs["num_of_loan"],
        "Delay_from_due_date": inputs["delay_from_due"],
        "Num_of_Delayed_Payment": inputs["num_delayed_payments"],
        "Changed_Credit_Limit": inputs["changed_credit_limit"],
        "Num_Credit_Inquiries": inputs["num_credit_inquiries"],
        "Outstanding_Debt": inputs["outstanding_debt"],
        "Credit_History_Age": inputs["credit_history_months"],
        "Total_EMI_per_month": inputs["total_emi"],
        "Amount_invested_monthly": inputs["amount_invested"],
        "Monthly_Balance": inputs["monthly_balance"],
        "Payment_of_Min_Amount": inputs["payment_of_min"],
        "Credit_Mix": inputs["credit_mix"],
        "Occupation": inputs["occupation"],
        "Payment_Behaviour": inputs["payment_behaviour"],
    }
    for lt in LOAN_TYPES:
        row[lt] = 1 if lt in inputs["loan_types"] else 0

    df = pd.DataFrame([row])
    if TRAIN_COLUMNS:
        df = df.reindex(columns=TRAIN_COLUMNS)
    return df


def predict(input_df: pd.DataFrame):
    label_idx = pipeline.predict(input_df)[0]
    label = LABEL_MAP.get(int(label_idx), str(label_idx))

    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(input_df)[0]
        except Exception:
            pass

    return label, proba


# Page config
st.set_page_config(page_title="Credit Score Predictor", layout="wide",)

st.title("Credit Score Prediction")
st.caption(f"Model: {BEST_MODEL_NAME}  |  Classes: Poor, Standard, Good")

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar - Input Form
with st.sidebar:
    st.header("Customer Data Input")

    with st.form("input_form"):
        st.subheader("Personal & Financial Info")
        age = st.number_input("Age", 18, 100, 35)
        occupation = st.selectbox("Occupation", OCCUPATIONS)
        annual_income = st.number_input("Annual Income (USD)", 0.0, 500_000.0, 50_000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 50_000.0, 4_000.0, step=100.0)

        st.subheader("Credit Accounts")
        num_bank_accounts = st.slider("Number of Bank Accounts", 0, 20, 3)
        num_credit_cards = st.slider("Number of Credit Cards", 0, 20, 2)
        num_of_loan = st.slider("Number of Loans", 0, 20, 2)
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])

        st.subheader("Payment Behaviour")
        interest_rate = st.slider("Interest Rate (%)", 0, 60, 15)
        delay_from_due = st.slider("Avg. Delay from Due Date (days)", 0, 100, 5)
        num_delayed_payments = st.slider("Number of Delayed Payments", 0, 50, 3)
        payment_of_min = st.selectbox("Pays Minimum Amount?", ["Yes", "No"])
        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOURS)

        st.subheader("Debt & Utilisation")
        outstanding_debt = st.number_input("Outstanding Debt (USD)", 0.0, 100_000.0, 1_000.0, step=100.0)
        changed_credit_limit = st.number_input("Changed Credit Limit (USD)", -10_000.0, 50_000.0, 0.0, step=100.0)
        num_credit_inquiries = st.slider("Number of Credit Inquiries", 0, 20, 2)

        st.subheader("History & EMI")
        credit_history_months = st.slider("Credit History Age (months)", 0, 500, 120)
        total_emi = st.number_input("Total EMI per Month (USD)", 0.0, 10_000.0, 300.0, step=50.0)
        amount_invested = st.number_input("Amount Invested Monthly (USD)", 0.0, 10_000.0, 200.0, step=50.0)
        monthly_balance = st.number_input("Monthly Balance (USD)", 0.0, 50_000.0, 500.0, step=50.0)

        st.subheader("Loan Types Held")
        loan_types = st.multiselect("Select all that apply", LOAN_TYPES, default=[])

        submitted = st.form_submit_button("Predict Credit Score", use_container_width=True)


# Prediction & Output
if submitted:
    inputs = dict(
        age=age, occupation=occupation,
        annual_income=annual_income, monthly_salary=monthly_salary,
        num_bank_accounts=num_bank_accounts, num_credit_cards=num_credit_cards,
        num_of_loan=num_of_loan, credit_mix=credit_mix,
        interest_rate=interest_rate, delay_from_due=delay_from_due,
        num_delayed_payments=num_delayed_payments, payment_of_min=payment_of_min,
        payment_behaviour=payment_behaviour, outstanding_debt=outstanding_debt,
        changed_credit_limit=changed_credit_limit,
        num_credit_inquiries=num_credit_inquiries, credit_history_months=credit_history_months,
        total_emi=total_emi, amount_invested=amount_invested,
        monthly_balance=monthly_balance, loan_types=loan_types,
    )

    input_df = build_input_df(inputs)
    label, proba = predict(input_df)
    color = LABEL_COLORS.get(label, "#888")

    st.session_state.history.append({**inputs, "Prediction": label, "Loan Types": ", ".join(loan_types)})

    # Result Banner
    st.markdown(
        f"""
        <div style='background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:20px; text-align:center; margin-bottom:20px'>
            <h1 style='color:{color}; margin:0'>Credit Score: {label}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Summary")
        summary = {
            "Age": age, "Occupation": occupation,
            "Annual Income": f"${annual_income:,.0f}",
            "Monthly Salary": f"${monthly_salary:,.0f}",
            "Credit Mix": credit_mix,
            "Outstanding Debt": f"${outstanding_debt:,.0f}",
            "Credit History": f"{credit_history_months} months",
            "Delayed Payments": num_delayed_payments,
        }
        summary_df = pd.DataFrame(summary.items(), columns=["Feature", "Value"]).astype(str)
        st.table(summary_df)

    with col2:
        if proba is not None:
            st.subheader("Prediction Probability")
            labels = [LABEL_MAP[i] for i in range(len(proba))]
            colors = [LABEL_COLORS.get(l, "#888") for l in labels]

            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(labels, proba * 100, color=colors, edgecolor="white")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Score Class Probabilities")
            for bar, p in zip(bars, proba):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{p*100:.1f}%", va="center", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
        else:
            st.info("Probability scores not available for this model.")

    # Feature Highlight Chart
    st.subheader("Key Financial Indicators")
    fin_metrics = {
        "Annual Income": annual_income / 1000,
        "Outstanding Debt": outstanding_debt / 1000,
        "Monthly Balance": monthly_balance / 1000,
        "Total EMI": total_emi,
        "Amount Invested": amount_invested,
    }
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.bar(fin_metrics.keys(), fin_metrics.values(), color="#4C72B0", edgecolor="white")
    ax2.set_ylabel("Value (USD / 1k for income & debt)")
    ax2.set_title("Financial Indicators Overview")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig2)


# Prediction History
if st.session_state.history:
    st.divider()
    st.subheader("Prediction History")
    df_hist = pd.DataFrame(st.session_state.history)
    display_cols = [c for c in ["age", "occupation", "annual_income", "credit_mix",
                                 "outstanding_debt", "Prediction"] if c in df_hist.columns]
    st.dataframe(df_hist[display_cols], use_container_width=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()