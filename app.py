import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from sentiment_plot import generate_sentiment_plot
from density_plot1 import generate_density_plot
from map_plot import generate_interactive_map

# =========================
# CONFIG (ONLY ONCE)
# =========================
st.set_page_config(page_title="Sentiment Dashboard + Chatbot", layout="wide")

# =========================
# DESCRIPTION
# =========================
desc = """📌 About This Dashboard

This dashboard presents an analysis of political sentiment in Tamil Nadu based on data collected from Twitter/X. The dataset was obtained using the official X API (paid tier), capturing approximately 4,000 recent public posts.

Each record includes tweet content, a unique tweet ID, sentiment classification, and geographic coordinates where available. The data was cleaned to remove duplicates and incomplete entries, ensuring only valid records with identifiable political targets and sentiment labels were retained.

Sentiment is categorized into four classes: positive, negative, neutral, and mixed.

Data collection was performed using the queries: ["TN Elections", "TN Polls"]. Since the X API prioritizes recent and relevant content for general queries, the dataset reflects current public discourse. However, it may still contain inherent platform-level or sampling biases typical of social media data.

At the moment This Dashboard Includes:

- Sentiment distribution
- Party-wise breakdown
- Geo visualization
- AI-powered tweet chatbot (RAG)
"""

st.title("📊 Tamil Nadu Political Sentiment Dashboard")
st.markdown(desc)

# =========================
# LOAD DATA (SHARED)
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("updated_with_coordinates.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['text'])
    return df

df = load_data()

# =========================
# VALIDATION
# =========================
required_cols = ['target_party', 'sentiment', 'latitude', 'longitude']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# =========================
# PARTY FILTER
# =========================
all_parties = sorted(df['target_party'].dropna().unique())

_, default_top5 = generate_sentiment_plot(df)

selected_parties = st.multiselect(
    "Select Parties",
    options=all_parties,
    default=default_top5
)

if not selected_parties:
    st.warning("Select at least one party")
    st.stop()

# =========================
# METRIC SELECT
# =========================
metric = st.selectbox(
    "Select Map Metric",
    ["total", "positive_pct", "negative_pct", "neutral_pct", "mixed_pct"]
)

# =========================
# ROW 1 → CHARTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Breakdown")
    fig1, _ = generate_sentiment_plot(df, selected_parties)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Sentiment Distribution")
    fig2 = generate_density_plot(df, selected_parties)
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# MAP
# =========================
st.subheader("📍 District Sentiment Map")

fig = generate_interactive_map(
    df,
    geojson_path="TAMIL NADU_DISTRICTS.geojson",
    selected_parties=selected_parties,
    metric=metric
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ===================== 🤖 CHATBOT ===========================
# ============================================================

st.markdown("---")
st.subheader("🧠 AI Tweet Assistant")

# =========================
# INIT MODELS (CACHE)
# =========================
@st.cache_resource
def load_models():
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return client, model

client, model = load_models()

INDEX_FILE = "faiss_index.index"
EMBED_FILE = "embeddings.npy"

# =========================
# BUILD / LOAD INDEX
# =========================
@st.cache_resource
def get_index(texts):

    if os.path.exists(INDEX_FILE) and os.path.exists(EMBED_FILE):
        index = faiss.read_index(INDEX_FILE)
        embeddings = np.load(EMBED_FILE)
        return index, embeddings

    st.info("Building vector index (one-time)...")

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(EMBED_FILE, embeddings)

    st.success("Index ready!")
    return index, embeddings

texts = df['text'].tolist()
index, embeddings = get_index(texts)

# =========================
# RETRIEVAL
# =========================
def retrieve(query, k=15):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, k)
    return df.iloc[I[0]]

# =========================
# GENERATE
# =========================
def generate_answer(query, retrieved_df):

    context = "\n".join(retrieved_df['text'].tolist())

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from tweets. No outside knowledge."
            },
            {
                "role": "user",
                "content": f"Tweets:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response.choices[0].message.content

# =========================
# CHAT STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# CHAT UI (CLEAN)
# =========================
chat_container = st.container()

with chat_container:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask about public sentiment, parties, trends...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Thinking..."):
            retrieved = retrieve(prompt)
            answer = generate_answer(prompt, retrieved)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)

        with st.expander("🔍 Retrieved Tweets"):
            st.dataframe(retrieved[['text']])