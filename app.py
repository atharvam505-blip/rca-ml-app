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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RCA Intelligence System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
section[data-testid="stSidebar"] *{color:#c9d1d9!important;}
.section-header{font-family:'IBM Plex Mono',monospace;font-size:.75rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:#58a6ff;border-left:3px solid #58a6ff;padding-left:.75rem;margin:2rem 0 1rem 0;}
.metric-card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:1.25rem;margin-bottom:.75rem;}
.metric-card .label{font-size:.7rem;font-family:'IBM Plex Mono',monospace;letter-spacing:.1em;color:#8b949e;text-transform:uppercase;}
.metric-card .value{font-size:1.6rem;font-weight:700;color:#f0f6fc;margin-top:.25rem;}
.chat-user{background:#1c2128;border:1px solid #30363d;border-radius:12px 12px 2px 12px;padding:.85rem 1.1rem;margin:.5rem 0 .5rem 20%;color:#e6edf3;font-size:.9rem;}
.chat-bot{background:#0f2444;border:1px solid #1f6feb;border-radius:2px 12px 12px 12px;padding:.85rem 1.1rem;margin:.5rem 20% .5rem 0;color:#cae8ff;font-size:.9rem;}
.chat-label{font-size:.65rem;font-family:'IBM Plex Mono',monospace;letter-spacing:.08em;color:#8b949e;margin-bottom:.3rem;text-transform:uppercase;}
.llm-box{background:#161b22;border:1px solid #30363d;border-left:4px solid #3fb950;border-radius:0 8px 8px 0;padding:1.5rem;color:#e6edf3;font-size:.88rem;line-height:1.7;white-space:pre-wrap;}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:20px;font-size:.7rem;font-family:'IBM Plex Mono',monospace;font-weight:600;letter-spacing:.05em;}
.badge-green{background:#1a4731;color:#3fb950;border:1px solid #3fb950;}
.badge-yellow{background:#3d2b00;color:#d29922;border:1px solid #d29922;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title, icon=""):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="IBM Plex Mono"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_data(file):
    for enc in ["utf-8", "latin-1", "cp1252", "utf-16"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file, encoding="latin-1", errors="replace")

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

# =============================================================================
# 2. ML MODELS
# =============================================================================
def run_ml_models(df, target_col, date_col=None):
    results = {}
    feature_cols = [c for c in df.columns if c != target_col and c != date_col]
    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Encode categoricals
    for col in X_raw.select_dtypes(include="object").columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    # Impute
    imputer = SimpleImputer(strategy="most_frequent")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols)

    # Random Forest
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
    results["is_regression"] = is_regression

    # KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = min(4, max(2, len(df) // 10))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    results["cluster_labels"] = labels
    results["n_clusters"] = k

    cluster_df = X.copy()
    cluster_df["Cluster"] = labels
    results["cluster_summary"] = cluster_df.groupby("Cluster").mean().round(2)

    # Time trend
    if date_col:
        try:
            trend_df = df[[date_col, target_col]].copy()
            trend_df[date_col] = pd.to_datetime(trend_df[date_col])
            trend_df = trend_df.sort_values(date_col)
            if pd.api.types.is_numeric_dtype(trend_df[target_col]):
                trend_df["period"] = trend_df[date_col].dt.to_period("M").astype(str)
                results["trend"] = trend_df.groupby("period")[target_col].mean().reset_index()
            else:
                results["trend"] = None
        except Exception:
            results["trend"] = None
    else:
        results["trend"] = None

    results["feature_cols"] = feature_cols
    return results

# =============================================================================
# 3. PROMPT ENGINEERING  (explicit template)
# =============================================================================
def generate_prompt(df, target_col, ml_results):
    top_feat = ml_results["top_features"]
    feature_lines = "\n".join(
        f"  {i+1}. {feat}: importance {score:.4f}"
        for i, (feat, score) in enumerate(top_feat.items())
    )
    cluster_lines = ml_results["cluster_summary"].to_string()
    trend_lines = ""
    if ml_results.get("trend") is not None:
        trend_lines = "\nTIME TREND (mean target per month):\n" + ml_results["trend"].tail(6).to_string(index=False)

    return textwrap.dedent(f"""
    You are a senior data analyst performing Root Cause Analysis.

    DATASET OVERVIEW:
    - Rows: {len(df)}  |  Columns: {list(df.columns)}
    - Target variable: {target_col}

    TOP FEATURE IMPORTANCES (Random Forest):
    {feature_lines}

    CLUSTER SEGMENTATION SUMMARY ({ml_results['n_clusters']} clusters via KMeans):
    {cluster_lines}
    {trend_lines}

    TASK:
    1. Identify the PRIMARY root cause of performance variation in "{target_col}".
    2. List the TOP 3 KEY DRIVERS with a one-sentence explanation for each.
    3. Provide 3 RECOMMENDED ACTIONS (specific and actionable).
    4. Flag any ANOMALIES or RISKS visible in the data.

    Keep response structured and concise. Use plain English. Use numbered lists only.
    """).strip()

# =============================================================================
# 4. LLM CALL
# =============================================================================
def call_llm(prompt, api_key, provider="claude"):
    if not api_key:
        return "⚠️  No API key provided. Insert your key in the sidebar."
    try:
        if provider == "claude" and ANTHROPIC_AVAILABLE:
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
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        elif provider == "groq" and OPENAI_AVAILABLE:
            # Groq uses OpenAI-compatible API — fast inference (LLaMA)
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        else:
            return "⚠️  Provider library not installed."
    except Exception as e:
        return f"❌  LLM error: {e}"

# =============================================================================
# 5. OCR
# =============================================================================
def run_ocr(image_file):
    if not OCR_AVAILABLE:
        return "pytesseract/Pillow not installed. Run: pip install pytesseract pillow"
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text.strip() or "No text detected."
    except Exception as e:
        return f"OCR error: {e}"

# =============================================================================
# CHARTS
# =============================================================================
def chart_feature_importance(importances):
    top = importances.head(10)
    colors = [f"rgba(88,166,255,{0.4+0.06*i})" for i in range(len(top))]
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=top.index[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.3f}" for v in top.values[::-1]], textposition="outside",
    ))
    fig.update_layout(title="Feature Importances (Random Forest)", **DARK_LAYOUT,
                      xaxis=dict(showgrid=False, color="#8b949e"),
                      yaxis=dict(showgrid=False, color="#c9d1d9"))
    return fig

def chart_clusters(df, labels, feature_cols):
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return None
    x_col, y_col = num_cols[0], num_cols[1]
    palette = ["#58a6ff","#3fb950","#d29922","#f78166"]
    fig = go.Figure()
    for cid in sorted(set(labels)):
        mask = labels == cid
        fig.add_trace(go.Scatter(
            x=df[x_col][mask], y=df[y_col][mask], mode="markers",
            name=f"Cluster {cid}",
            marker=dict(size=7, color=palette[cid % 4], opacity=0.75),
        ))
    fig.update_layout(title="KMeans Segments", **DARK_LAYOUT,
                      xaxis=dict(title=x_col, showgrid=False, color="#8b949e"),
                      yaxis=dict(title=y_col, showgrid=False, color="#c9d1d9"))
    return fig

def chart_trend(trend_df, target_col):
    fig = go.Figure(go.Scatter(
        x=trend_df["period"], y=trend_df[target_col],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=6, color="#58a6ff"),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))
    fig.update_layout(title=f"Time Trend – {target_col}", **DARK_LAYOUT,
                      xaxis=dict(showgrid=False, color="#8b949e"),
                      yaxis=dict(showgrid=False, color="#c9d1d9"))
    return fig

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Session state
    for key, default in [("chat_history", []), ("ml_results", None),
                          ("rca_text", ""), ("df", None),
                          ("target_col", ""), ("date_col", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 RCA System")
        st.markdown("---")
        st.markdown("### LLM Configuration")
        provider = st.selectbox("Provider", ["claude", "openai", "groq"])
        api_key = st.text_input("API Key", type="password",
                                placeholder="sk-... or anthropic key")
        st.markdown("---")
        st.markdown("### Status")
        ocr_label = "green" if OCR_AVAILABLE else "yellow"
        ocr_txt = "OCR ✓" if OCR_AVAILABLE else "OCR unavailable"
        llm_ok = ANTHROPIC_AVAILABLE or OPENAI_AVAILABLE
        llm_label = "green" if llm_ok else "yellow"
        llm_txt = "LLM ✓" if llm_ok else "LLM unavailable"
        st.markdown(f'<span class="badge badge-{ocr_label}">{ocr_txt}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge badge-{llm_label}">{llm_txt}</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Multimodal RCA System v1.0")

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:1.5rem 0 .5rem 0;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.2em;color:#58a6ff;text-transform:uppercase;">Multimodal Intelligence Platform</div>
      <div style="font-size:2rem;font-weight:700;color:#f0f6fc;margin-top:.2rem;">Root Cause Analysis System</div>
      <div style="font-size:.9rem;color:#8b949e;margin-top:.3rem;">ML-powered insights · LLM explanations · OCR · Conversational interface</div>
    </div>
    <hr style="border-color:#21262d;margin:1rem 0;">
    """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 1 – UPLOAD & CONTROLS
    # ═════════════════════════════════════════════════════════════════════════
    section("01 · Data Input", "📂")

    col_upload, col_controls = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        if st.button("⚡ Load demo dataset", use_container_width=True):
            np.random.seed(42)
            n = 300
            demo = pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
                "region": np.random.choice(["North","South","East","West"], n),
                "product": np.random.choice(["A","B","C"], n),
                "sales_rep_exp_years": np.random.randint(1, 15, n),
                "marketing_spend": np.random.exponential(5000, n).round(0),
                "competitor_price": np.random.normal(100, 15, n).round(1),
                "customer_satisfaction": np.random.uniform(3, 5, n).round(1),
                "revenue": (np.random.normal(50000,8000,n) + np.random.choice([0,-10000],n,p=[.8,.2])).round(0),
            })
            st.session_state.df = demo
            st.success("Demo dataset loaded – 300 rows, 8 columns.")

    with col_controls:
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)

        df = st.session_state.df
        run_btn = False

        if df is not None:
            date_cols, cat_cols, num_cols = detect_columns(df)
            target_col = st.selectbox("🎯 Target column", df.columns.tolist(),
                                      index=len(df.columns)-1)
            date_col = st.selectbox("📅 Date column (optional)",
                                    ["None"] + date_cols + cat_cols)
            date_col = None if date_col == "None" else date_col
            run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    # Data preview
    df = st.session_state.df
    if df is not None:
        with st.expander("📋 Data preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="label">Rows</div><div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="label">Columns</div><div class="value">{len(df.columns)}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="label">Missing %</div><div class="value">{df.isnull().mean().mean()*100:.1f}%</div></div>', unsafe_allow_html=True)

    # Run analysis
    if df is not None and run_btn:
        with st.spinner("Running ML models…"):
            try:
                ml = run_ml_models(df, target_col, date_col)
                st.session_state.ml_results = ml
                st.session_state.target_col = target_col
                st.session_state.date_col = date_col
                prompt = generate_prompt(df, target_col, ml)
                st.session_state.rca_text = call_llm(prompt, api_key, provider)
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    ml_results = st.session_state.ml_results
    target_col_s = st.session_state.target_col
    date_col_s = st.session_state.date_col

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 2 – CHARTS
    # ═════════════════════════════════════════════════════════════════════════
    if ml_results:
        section("02 · Visual Analysis", "📊")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Cluster Map", "Time Trend"])

        with tab1:
            st.plotly_chart(chart_feature_importance(ml_results["importances"]),
                            use_container_width=True)
        with tab2:
            feat_cols = ml_results.get("feature_cols", [])
            fig2 = chart_clusters(df, ml_results["cluster_labels"], feat_cols)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Need ≥2 numeric feature columns for cluster scatter.")
        with tab3:
            trend = ml_results.get("trend")
            if trend is not None:
                st.plotly_chart(chart_trend(trend, target_col_s), use_container_width=True)
            else:
                st.info("No date column selected or target is non-numeric.")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 – LLM ROOT CAUSE EXPLANATION
    # ═════════════════════════════════════════════════════════════════════════
    if st.session_state.rca_text:
        section("03 · Root Cause Explanation", "🧠")
        st.markdown(f'<div class="llm-box">{st.session_state.rca_text}</div>',
                    unsafe_allow_html=True)
        with st.expander("🔎 View raw prompt", expanded=False):
            if ml_results and df is not None:
                st.code(generate_prompt(df, target_col_s, ml_results), language="text")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 4 – CHAT
    # ═════════════════════════════════════════════════════════════════════════
    section("04 · Conversational Interface", "💬")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-label">You</div><div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label">RCA Assistant</div><div class="chat-bot">{msg["content"]}</div>',
                        unsafe_allow_html=True)

    user_q = st.chat_input("Ask about the data… e.g. 'Why is performance dropping?'")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        ctx = ""
        if ml_results:
            top = ", ".join(f"{k}({v:.3f})" for k, v in list(ml_results["top_features"].items())[:5])
            ctx = f"\nCONTEXT – Top drivers: {top}\n"
            if st.session_state.rca_text:
                ctx += f"Prior RCA: {st.session_state.rca_text[:400]}…\n"
        chat_prompt = f"You are an RCA assistant. Answer concisely (3-5 sentences).\n{ctx}\nQUESTION: {user_q}"
        with st.spinner("Thinking…"):
            reply = call_llm(chat_prompt, api_key, provider)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 – IMAGE / OCR
    # ═════════════════════════════════════════════════════════════════════════
    section("05 · Image Analysis (OCR)", "🖼")

    col_img, col_ocr = st.columns([1, 1], gap="large")
    with col_img:
        uploaded_image = st.file_uploader("Upload image (chart, report, whiteboard…)",
                                          type=["png","jpg","jpeg","tiff","bmp"],
                                          key="img_up")
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

    with col_ocr:
        if uploaded_image:
            with st.spinner("Running OCR…"):
                extracted = run_ocr(uploaded_image)
            st.markdown("**Extracted text:**")
            st.code(extracted or "(empty)", language="text")
            if extracted and st.button("🧠 Explain via LLM", use_container_width=True):
                ocr_prompt = textwrap.dedent(f"""
                Text extracted from an image:
                ---
                {extracted}
                ---
                1. Summarise what this content represents.
                2. Identify key metrics, trends, or anomalies.
                3. Suggest one actionable insight.
                Be concise (5-7 sentences).
                """).strip()
                with st.spinner("Generating explanation…"):
                    ocr_resp = call_llm(ocr_prompt, api_key, provider)
                st.markdown(f'<div class="llm-box">{ocr_resp}</div>', unsafe_allow_html=True)
        else:
            st.info("Upload an image to extract text and analyse it with the LLM.")

    st.markdown("""
    <hr style="border-color:#21262d;margin:3rem 0 1rem 0;">
    <div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#30363d;letter-spacing:.15em;">
      MULTIMODAL RCA SYSTEM · STREAMLIT · ML + LLM + OCR
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
