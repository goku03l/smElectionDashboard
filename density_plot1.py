import pandas as pd
import plotly.express as px

def generate_density_plot(df, selected_parties=None):
    df = df.copy()

    # ---------------------------
    # SENTIMENT → NUMERIC
    # ---------------------------
    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1,
        "mixed": 0.5
    }

    df['sentiment_clean'] = df['sentiment'].str.lower().str.strip()
    df['sentiment_score'] = df['sentiment_clean'].map(sentiment_map)

    df = df.dropna(subset=['sentiment_score'])

    # ---------------------------
    # FILTER PARTIES
    # ---------------------------
    if selected_parties:
        df = df[df['target_party'].isin(selected_parties)]

    # ---------------------------
    # DENSITY / DISTRIBUTION PLOT
    # ---------------------------
    fig = px.histogram(
        df,
        x="sentiment_score",
        color="target_party",
        marginal="box",   # gives density feel
        nbins=30,
        barmode="overlay",
        opacity=0.5
    )

    fig.update_layout(
        title="Sentiment Density Distribution",
        xaxis_title="Sentiment Score (-1 to +1)",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    return fig