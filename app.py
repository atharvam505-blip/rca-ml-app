"""
Multimodal Conversational Root Cause Analysis System
=====================================================
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


st.set_page_config(
    page_title="RCA Intelligence System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            except Exception:
                cat_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        else:
            num_cols.append(col)
    return date_cols, cat_cols, num_cols


def run_ml_models(df, target_col, date_col=None):
    results = {}
    feature_cols = [c for c in df.columns if c != target_col and c != date_col]
    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    for col in X_raw.select_dtypes(include="object").columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    imputer = SimpleImputer(strategy="most_frequent")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols)

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

    return results


def generate_prompt(df, target_col, ml_results):
    return f"Analyze root causes for {target_col} using ML insights."


# ✅ UPDATED LLM FUNCTION
def call_llm(prompt, api_key, provider="claude"):
    if not api_key:
        return "⚠️ No API key provided."

    try:
        if provider == "claude" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=500,
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

        # ✅ GROQ ADDED HERE
        elif provider == "groq" and GROQ_AVAILABLE:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a data analyst."},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content

        else:
            return "Provider not available."

    except Exception as e:
        return f"Error: {e}"


def main():
    st.title("RCA System")

    # ✅ UPDATED DROPDOWN
    provider = st.selectbox("Provider", ["groq", "claude", "openai"])
    api_key = st.text_input("API Key")

    file = st.file_uploader("Upload CSV")

    if file:
        df = load_data(file)
        target = st.selectbox("Target", df.columns)

        if st.button("Run"):
            ml = run_ml_models(df, target)
            prompt = generate_prompt(df, target, ml)
            result = call_llm(prompt, api_key, provider)
            st.write(result)


if __name__ == "__main__":
    main()
