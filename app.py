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
# CONFIG
# =========================
st.set_page_config(page_title="Social Media Election Analysis", layout="wide")

# =========================
# DESCRIPTION
# =========================
desc = """
📌 **About This Dashboard**

This dashboard presents an analysis of political sentiment in Tamil Nadu based on data collected from Twitter/X. The dataset reflects recent public discourse around elections and political activity, offering insights into how people are reacting online. While it provides a useful snapshot, it is important to note that social media data may include inherent platform and sampling biases.

---

🔍 **Key Details**

- **Data Source:**  
  Collected using the official X (Twitter) API (paid tier)

- **Dataset Size:**  
  Approximately 4,000 recent public posts ( after data cleaning reduced to 3369 ).

- **Sentiment Categories:**  
  - Positive  
  - Negative  
  - Neutral  
  - Mixed  

- **Search Queries Used:**  
  - "TN Elections"  
  - "TN Polls"  
  
- **Includes:**
    - Sentiment distribution
    - Party-wise breakdown
    - Geo visualization
    - AI-powered tweet chatbot (RAG)
    
- **Important Note:**  
   - The X API prioritises recent and relevant content for general queries, so the dataset reflects current trends. However, it may still include biases typical of social media platforms.
   - This project is done with the intention of learning and implementing something from scratch with no lying intention,this topic has been taken because Elections are most happening topic at the moment
"""

st.title("📊 Tamil Nadu Political Sentiment Dashboard")
st.markdown(desc)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("updated_with_coordinates.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['text'])
    df = df.reset_index(drop=True)
    return df

df = load_data()

# 🔐 LOCK DATA FOR CHATBOT (CRITICAL FIX)
df_chat = df.copy()

# =========================
# VALIDATION
# =========================
required_cols = ['target_party', 'sentiment', 'latitude', 'longitude']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# =========================
# PARTY FILTER (UI ONLY)
# =========================
all_parties = sorted(df['target_party'].dropna().unique())

_, default_top5 = generate_sentiment_plot(df.copy())

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
# CHART DATA (FILTERED COPY)
# =========================
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["target_party"].isin(selected_parties)]

# =========================
# ROW 1 → CHARTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Breakdown")
    fig1, _ = generate_sentiment_plot(df_filtered.copy(), selected_parties)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Sentiment Distribution")
    fig2 = generate_density_plot(df_filtered.copy(), selected_parties)
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# MAP
# =========================
st.subheader("📍 District Sentiment Map")

fig = generate_interactive_map(
    df_filtered.copy(),
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
# INIT MODELS
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
# BUILD / LOAD INDEX (ONLY ON df_chat)
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

texts = df_chat['text'].tolist()
index, embeddings = get_index(texts)

# =========================
# SAFE RETRIEVAL (FIXED)
# =========================
def retrieve(query, k=15):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, k)

    # 🔒 CRITICAL FIX: validate indices
    valid_indices = [i for i in I[0] if 0 <= i < len(df_chat)]

    if not valid_indices:
        return pd.DataFrame()

    return df_chat.iloc[valid_indices]

# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query, retrieved_df):

    if retrieved_df.empty:
        return "I couldn't find relevant tweets for that query."

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
# CHAT UI
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
