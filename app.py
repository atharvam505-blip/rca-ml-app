"""
Multimodal Conversational Root Cause Analysis System
Run:  streamlit run app.py
"""

import os
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ── Optional libs ─────────────────────────────────────────────────────────────
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ✅ GROQ ADDED
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RCA Intelligence System",
    page_icon="🔍",
    layout="wide",
)

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except:
        return pd.read_csv(file, encoding='latin1')


def detect_columns(df):
    date_cols, cat_cols, num_cols = [], [], []
    for col in df.columns:
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].dropna().head(20))
                date_cols.append(col)
            except:
                cat_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        else:
            num_cols.append(col)
    return date_cols, cat_cols, num_cols


# =============================================================================
# ML MODELS
# =============================================================================
def run_ml_models(df, target_col, date_col=None):
    results = {}
    feature_cols = [c for c in df.columns if c != target_col and c != date_col]

    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    for col in X_raw.select_dtypes(include="object").columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    X = pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(X_raw),
                     columns=feature_cols)

    is_regression = pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10

    if is_regression:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        y_raw = LabelEncoder().fit_transform(y_raw.astype(str))
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y_raw)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    results["importances"] = importances
    results["top_features"] = importances.head(10)

    # Clustering
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    results["cluster_labels"] = labels
    results["cluster_summary"] = pd.DataFrame(X).assign(cluster=labels).groupby("cluster").mean()

    return results


# =============================================================================
# PROMPT
# =============================================================================
def generate_prompt(df, target_col, ml_results):
    features = "\n".join([f"{k}: {v:.3f}" for k, v in ml_results["top_features"].items()])
    return f"""
You are a data analyst performing root cause analysis.

Target: {target_col}

Top Features:
{features}

Tasks:
1. Identify root cause
2. Key drivers
3. Actions
"""


# =============================================================================
# LLM (UPDATED WITH GROQ)
# =============================================================================
def call_llm(prompt, api_key, provider="groq"):

    if not api_key:
        return "⚠️ No API key provided"

    try:
        if provider == "groq" and GROQ_AVAILABLE:
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        elif provider == "claude" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        elif provider == "openai" and OPENAI_AVAILABLE:
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

        else:
            return "Provider not available"

    except Exception as e:
        return str(e)


# =============================================================================
# MAIN APP
# =============================================================================
def main():

    st.title("🔍 Root Cause Analysis System")

    # Sidebar
    provider = st.sidebar.selectbox("LLM Provider", ["groq", "claude", "openai"])
    api_key = st.sidebar.text_input("API Key", type="password")

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file:
        df = load_data(uploaded_file)

        target_col = st.selectbox("Target", df.columns)

        if st.button("Run Analysis"):

            ml_results = run_ml_models(df, target_col)

            st.subheader("Feature Importance")
            st.bar_chart(ml_results["importances"])

            prompt = generate_prompt(df, target_col, ml_results)

            with st.spinner("Generating insights..."):
                output = call_llm(prompt, api_key, provider)

            st.subheader("Root Cause Analysis")
            st.write(output)


if __name__ == "__main__":
    main()
