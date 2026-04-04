import pandas as pd
import numpy as np
import plotly.express as px
import json


def generate_interactive_map(df, geojson_path, selected_parties=None, metric="total"):

    df = df.copy()

    # ---------------------------
    # FILTER PARTIES
    # ---------------------------
    if selected_parties:
        df = df[df["target_party"].isin(selected_parties)]

    # ---------------------------
    # ⚠️ IMPORTANT: DISTRICT COLUMN REQUIRED
    # ---------------------------
    if "dtname" not in df.columns:
        raise ValueError("Your dataframe must contain a 'dtname' column (district name)")

    df["dtname"] = df["dtname"].str.lower().str.strip()

    # ---------------------------
    # LOAD GEOJSON
    # ---------------------------
    with open(geojson_path) as f:
        geojson = json.load(f)

    # Normalize geojson district names
    for feature in geojson["features"]:
        feature["properties"]["dtname"] = (
            feature["properties"]["dtname"].lower().strip()
        )

    # ---------------------------
    # AGGREGATIONS
    # ---------------------------
    sentiment_counts = (
        df.groupby(["dtname", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all sentiment columns exist
    for col in ["positive", "negative", "neutral", "mixed"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0

    sentiment_counts["total"] = (
        sentiment_counts["positive"]
        + sentiment_counts["negative"]
        + sentiment_counts["neutral"]
        + sentiment_counts["mixed"]
    )

    # Percentages
    for col in ["positive", "negative", "neutral", "mixed"]:
        sentiment_counts[f"{col}_pct"] = (
            sentiment_counts[col] / sentiment_counts["total"].replace(0, np.nan)
        ) * 100

    sentiment_counts = sentiment_counts.fillna(0)

    # ---------------------------
    # LOG SCALE
    # ---------------------------
    if metric == "total":
        sentiment_counts["plot_value"] = np.log1p(sentiment_counts["total"])
    else:
        sentiment_counts["plot_value"] = sentiment_counts[metric]

    # ---------------------------
    # MAP
    # ---------------------------
    fig = px.choropleth_mapbox(
        sentiment_counts,
        geojson=geojson,
        locations="dtname",
        featureidkey="properties.dtname",
        color="plot_value",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=6,
        center={"lat": 11.1271, "lon": 78.6569},
        opacity=0.75,
        hover_data={
            "dtname": True,
            "total": True,
            "positive_pct": ":.1f",
            "negative_pct": ":.1f",
            "neutral_pct": ":.1f",
            "mixed_pct": ":.1f",
            "plot_value": False,
        },
    )

    # ---------------------------
    # OPTIONAL POINTS (no geopandas)
    # ---------------------------
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])

        fig.add_scattermapbox(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name="Tweets",
        )

    # ---------------------------
    # LAYOUT
    # ---------------------------
    fig.update_layout(
        title="Tamil Nadu Sentiment Map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    return fig