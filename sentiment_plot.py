import pandas as pd
import plotly.graph_objects as go


def generate_sentiment_plot(df, selected_parties=None):
    # ---------------------------
    # CLEAN
    # ---------------------------
    df = df.dropna(subset=['target_party', 'sentiment'])
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

    sentiments = ['positive', 'neutral', 'negative', 'mixed']

    # ---------------------------
    # DEFAULT TOP 5
    # ---------------------------
    top_parties = df['target_party'].value_counts().head(5).index.tolist()

    if not selected_parties:
        selected_parties = top_parties

    df_filtered = df[df['target_party'].isin(selected_parties)]

    # ---------------------------
    # GROUP COUNTS
    # ---------------------------
    grouped = (
        df_filtered.groupby(['target_party', 'sentiment'])
        .size()
        .unstack(fill_value=0)
    )

    # Sort by total descending
    grouped['total'] = grouped.sum(axis=1)
    grouped = grouped.sort_values(by='total', ascending=False)
    grouped = grouped.drop(columns=['total'])

    # Ensure all sentiment columns exist
    for s in sentiments:
        if s not in grouped.columns:
            grouped[s] = 0

    grouped = grouped[sentiments]

    # ---------------------------
    # PERCENTAGE FOR ALL PARTIES
    # ---------------------------
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # ---------------------------
    # COLORS
    # ---------------------------
    colors = {
        'positive': 'green',
        'negative': 'red',
        'neutral': 'orange',
        'mixed': 'blue'
    }

    # ---------------------------
    # PLOT
    # ---------------------------
    fig = go.Figure()

    for sentiment in sentiments:
        fig.add_trace(go.Bar(
            x=grouped.index,
            y=grouped[sentiment],
            name=sentiment.capitalize(),
            marker_color=colors[sentiment],
            text=[f"{pct.loc[party, sentiment]:.1f}%" for party in grouped.index],
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        title="Sentiment Count (All parties show %)",
        xaxis_title="Target Party",
        yaxis_title="Tweet Count",
        legend_title="Sentiment",
        template="plotly_white"
    )

    return fig, top_parties