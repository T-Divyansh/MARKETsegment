import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and data
filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# App design
st.set_page_config(page_title="Market Segment Predictor", layout="centered")
st.title(" Market Segment Prediction")

# Form for user input
with st.form("prediction_form"):
    st.header("Enter Customer Details:")
    balance = st.number_input('Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input('Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input('Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input('OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input('Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input('Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input('Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input('Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input('Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input('Cash Advance Trx', step=1)
    purchases_trx = st.number_input('Purchases Trx', step=1)
    credit_limit = st.number_input('Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input('Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input('Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input('Tenure', step=1)

    submitted = st.form_submit_button("Submit")

# On submit
if submitted:
    data = [[
        balance, balance_frequency, purchases, oneoff_purchases, installments_purchases,
        cash_advance, purchases_frequency, oneoff_purchases_frequency,
        purchases_installment_frequency, cash_advance_frequency, cash_advance_trx,
        purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure
    ]]

    cluster = loaded_model.predict(data)[0]
    st.success(f" This customer belongs to Cluster {cluster}")

    # Filter the dataset
    cluster_df = df[df['Cluster'] == cluster]

    st.subheader(f" Feature Distributions for Cluster {cluster}")
    
    # Plot feature distributions
    features = cluster_df.drop(columns=['Cluster']).columns
    cols = st.columns(4) 

    for i, feature in enumerate(features):
        with cols[i % len(cols)]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(cluster_df[feature], ax=ax, kde=True, bins=20, color='darkblue')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            st.pyplot(fig)

