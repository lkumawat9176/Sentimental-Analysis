import streamlit as st
import pandas as pd
from transformers import pipeline
from typing import List
import io
import matplotlib.pyplot as plt

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="SentimentScope — Smart Text Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM DARK THEME ----
st.markdown("""
    <style>
        body { background-color: #1e1e1e; color: #ffffff; }
        .main { background-color: #2e2e2e; padding: 20px 40px; border-radius: 15px; }
        h1, h2, h3, h4 { color: #ffffff; }
        .stDataFrame { border-radius: 10px; background-color: #3a3a3a; color: #ffffff; }
        .stButton>button {
            background-color: #4b8bbe;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1.5em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #3b7bb0;
        }
        .metric-box {
            background-color: #3a3a3a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 10px;
            color: #ffffff;
        }
        textarea, input {
            background-color: #3a3a3a;
            color: #ffffff;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.title("SentimentScope")
st.subheader("Analyze reviews, comments, or social media posts — powered by Transformers")
st.markdown("---")

# ---- LOAD MODEL ----
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis", truncation=True)

# ---- HELPERS ----
def clean_text_basic(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()

def aspect_tags(text: str, aspects: List[str]) -> List[str]:
    found = []
    t = text.lower()
    for a in aspects:
        if a.lower() in t:
            found.append(a)
    return found if found else ["general"]

def compute_net_sentiment(df: pd.DataFrame) -> float:
    if 'label' not in df.columns or len(df) == 0:
        return 0.0
    pos = df['label'].str.contains('POS', case=False).sum()
    neg = df['label'].str.contains('NEG', case=False).sum()
    total = len(df)
    return round((pos - neg) / total * 100, 2) if total > 0 else 0.0

# ---- SIDEBAR ----
st.sidebar.header("Configuration Panel")
sample_mode = st.sidebar.checkbox("Use sample dataset", value=True)
show_aspect_input = st.sidebar.checkbox("Enable aspect keyword detection", value=True)

sample_csv = """text,created_at,source
"I love the coffee and the ambience at BlueBean!",2025-10-01T10:00:00,Tweet
"Terrible service today — waited 45 mins and got the wrong order.",2025-10-02T14:21:00,Review
"Okay experience, pastries were fine.",2025-09-30T08:12:00,Review
"Best pastries in town! Highly recommend.",2025-10-03T09:45:00,Tweet
"Not happy with the new parking policy.",2025-10-04T12:30:00,Comment
"Menu needs more vegan options",2025-09-28T15:00:00,Review
"Staff were very polite and helpful.",2025-10-06T11:20:00,Review
"Food was cold when delivered.",2025-10-07T19:10:00,Comment
"""

uploaded = st.sidebar.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])
if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df_in = pd.DataFrame(columns=["text", "created_at", "source"])
else:
    if sample_mode:
        df_in = pd.read_csv(io.StringIO(sample_csv))
    else:
        st.sidebar.info("Upload a CSV or enable sample dataset.")
        df_in = pd.DataFrame(columns=["text", "created_at", "source"])

default_aspects = ["service", "food", "price", "parking", "staff", "ambience", "delivery"]
aspects_input = st.sidebar.text_input("Aspect keywords (comma-separated)", ",".join(default_aspects))
aspects = [a.strip() for a in aspects_input.split(",") if a.strip()]

st.markdown("### Data Preview")
st.dataframe(df_in.head(10), use_container_width=True)

# ---- RUN SENTIMENT ANALYSIS ----
if st.button("Run Sentiment Analysis"):
    if "text" not in df_in.columns:
        st.error("CSV must include a 'text' column.")
    else:
        with st.spinner("Analyzing sentiments... Please wait."):
            nlp = load_pipeline()
            texts = df_in['text'].astype(str).tolist()
            cleaned = [clean_text_basic(t) for t in texts]

            try:
                raw_preds = nlp(cleaned, truncation=True)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

            preds = []
            for r in raw_preds:
                if isinstance(r, list):
                    item = r[0] if len(r) > 0 and isinstance(r[0], dict) else {"label": "UNKNOWN", "score": 0.0}
                elif isinstance(r, dict):
                    item = r
                else:
                    item = {"label": str(r), "score": 0.0}
                preds.append({"label": item.get("label", "UNKNOWN"), "score": float(item.get("score", 0.0))})

            results = []
            for orig, p in zip(texts, preds):
                result = {
                    "text": orig,
                    "label": p["label"],
                    "score": p["score"],
                    "aspect": ",".join(aspect_tags(orig, aspects))
                }
                results.append(result)

            df_results = pd.DataFrame(results)
            df_results = df_results.drop(columns=["text"], errors="ignore")
            final_df = pd.concat([df_in.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)

            st.success("Sentiment analysis completed successfully!")

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='metric-box'><h3>Total Entries</h3><h2>{len(final_df)}</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-box'><h3>Net Sentiment</h3><h2>{compute_net_sentiment(final_df)}%</h2></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-box'><h3>Unique Aspects</h3><h2>{len(aspects)}</h2></div>", unsafe_allow_html=True)

            st.markdown("### Sentiment Distribution")
            dist = final_df['label'].value_counts()
            fig, ax = plt.subplots()
            dist.plot(kind="bar", ax=ax, color=['#3498db', '#e74c3c', '#f1c40f'])
            ax.set_xlabel("Sentiment Label")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Breakdown")
            st.pyplot(fig)

            if show_aspect_input:
                st.markdown("### Aspect-Based Breakdown")
                aspect_counts = final_df.copy()
                aspect_counts['aspect'] = aspect_counts['aspect'].astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
                exploded = aspect_counts.explode('aspect')
                table = exploded.groupby('aspect')['label'].value_counts().unstack(fill_value=0)
                st.dataframe(table, use_container_width=True)

# ---- SINGLE TEXT CHECK ----
st.sidebar.markdown("---")
st.sidebar.markdown("*Quick single-text sentiment check*")
single_text = st.sidebar.text_area("Paste text here to analyze", "")
if st.sidebar.button("Check sentiment"):
    if single_text.strip() == "":
        st.sidebar.warning("Enter some text to analyze.")
    else:
        try:
            nlp2 = load_pipeline()
            out = nlp2(single_text, truncation=True)
        except Exception as e:
            st.sidebar.error(f"Model error: {e}")
            out = None

        if out:
            o = out[0] if isinstance(out, list) and len(out) > 0 else out
            if isinstance(o, list) and len(o) > 0 and isinstance(o[0], dict):
                o = o[0]
            if isinstance(o, dict):
                st.sidebar.success(f"Label: {o.get('label')} | Score: {round(float(o.get('score', 0.0)), 3)}")
            else:
                st.sidebar.info(str(o))
