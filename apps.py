import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Load saved objects
xgb_model = joblib.load("xgb_model.pkl")
gene_list = joblib.load("gene_list.pkl")
shap_df = joblib.load("shap_genes.pkl")

st.title("AI-Based Gene Expression Classifier")
st.write("Demo app for melanoma vs benign classification using RNA-seq data.")

st.header("Input Sample Expression Values")

input_data = []

for gene in gene_list[:10]:  # only first 10 genes for demo
    value = st.slider(f"{gene}", -3.0, 3.0, 0.0)
    input_data.append(value)

# Pad remaining genes with zeros
input_data = input_data + [0] * (len(gene_list) - len(input_data))

sample = pd.DataFrame([input_data], columns=gene_list)

if st.button("Predict Sample"):
    prob = xgb_model.predict_proba(sample)[0][1]

    if prob >= 0.5:
        st.success(f"Predicted Class: Melanoma (Probability = {prob:.2f})")
    else:
        st.success(f"Predicted Class: Benign (Probability = {1 - prob:.2f})")

st.header("Top Influential Genes (SHAP)")

st.dataframe(
    shap_df.head(10)[["Gene_ID", "Mean_SHAP"]]
)

st.info(
    "SHAP values indicate how much each gene contributes to the model's prediction. "
    "Higher SHAP values indicate stronger influence."
)

