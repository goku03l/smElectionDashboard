import pandas as pd
import geopandas as gpd
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

    # Remove missing coords
    df = df.dropna(subset=["latitude", "longitude"])

    # ---------------------------
    # GEO POINTS
    # ---------------------------
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )

    # ---------------------------
    # LOAD DISTRICTS
    # ---------------------------
    gdf_districts = gpd.read_file(geojson_path)
    gdf_districts["dtname"] = gdf_districts["dtname"].str.lower().str.strip()

    # CRS match
    gdf_points = gdf_points.to_crs(gdf_districts.crs)

    # ---------------------------
    # SPATIAL JOIN
    # ---------------------------
    gdf_joined = gpd.sjoin(
        gdf_points,
        gdf_districts,
        how="left",
        predicate="within"
    )

    gdf_joined["dtname"] = gdf_joined["dtname"].str.lower().str.strip()

    # ---------------------------
    # AGGREGATIONS
    # ---------------------------
    sentiment_counts = (
        gdf_joined
        .groupby(["dtname", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all sentiment columns exist
    for col in ["positive", "negative", "neutral", "mixed"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0

    sentiment_counts["total"] = (
        sentiment_counts["positive"] +
        sentiment_counts["negative"] +
        sentiment_counts["neutral"] +
        sentiment_counts["mixed"]
    )

    # Percentages
    for col in ["positive", "negative", "neutral", "mixed"]:
        sentiment_counts[f"{col}_pct"] = (
            sentiment_counts[col] / sentiment_counts["total"].replace(0, np.nan)
        ) * 100

    sentiment_counts = sentiment_counts.fillna(0)

    # ---------------------------
    # MERGE WITH DISTRICTS
    # ---------------------------
    merged = gdf_districts.merge(
        sentiment_counts,
        on="dtname",
        how="left"
    )

    merged = merged.fillna(0)

    # ---------------------------
    # LOG SCALE FOR TOTAL
    # ---------------------------
    if metric == "total":
        merged["plot_value"] = np.log1p(merged["total"])
    else:
        merged["plot_value"] = merged[metric]

    # ---------------------------
    # GEOJSON
    # ---------------------------
    geojson = json.loads(merged.to_json())

    # ---------------------------
    # MAP
    # ---------------------------
    fig = px.choropleth_mapbox(
        merged,
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
            "plot_value": False
        }
    )

    # ---------------------------
    # ADD POINTS (OPTIONAL BUT NICE)
    # ---------------------------
    fig.add_scattermapbox(
        lat=gdf_points.geometry.y,
        lon=gdf_points.geometry.x,
        mode="markers",
        marker=dict(size=5, opacity=0.35),
        name="Tweets"
    )

    # ---------------------------
    # LAYOUT
    # ---------------------------
    fig.update_layout(
        title="Tamil Nadu Sentiment Map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    return fig