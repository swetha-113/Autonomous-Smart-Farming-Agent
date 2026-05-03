"""
src/utils/visualizations.py
Plotly chart helpers for the Streamlit dashboard.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PALETTE = {
    "primary":   "#2ECC71",
    "danger":    "#E74C3C",
    "warning":   "#F39C12",
    "info":      "#3498DB",
    "dark":      "#1A1A2E",
    "surface":   "#16213E",
    "card":      "#0F3460",
    "text":      "#E8F5E9",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def disease_confidence_chart(top3: list) -> go.Figure:
    """Horizontal bar chart for top-3 disease predictions."""
    labels = [d["class"].replace("___", " — ").replace("_", " ") for d in top3]
    values = [d["confidence"] * 100 for d in top3]
    colors = [PALETTE["danger"], PALETTE["warning"], PALETTE["info"]]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors[:len(labels)],
        text=[f"{v:.1f}%" for v in values],
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(color="white", size=13)
    ))
    fig.update_layout(
        title="Disease Detection Confidence",
        xaxis=dict(range=[0, 100], title="Confidence (%)", gridcolor="#2a3a5c"),
        yaxis=dict(autorange="reversed", gridcolor="#2a3a5c"),
        **CHART_LAYOUT
    )
    return fig


def soil_radar_chart(params: dict, health_scores: dict) -> go.Figure:
    """Radar chart comparing actual vs optimal soil parameters."""
    from config import OPTIMAL_RANGES
    cats = [p for p in params if p in OPTIMAL_RANGES]
    if not cats:
        return go.Figure()

    # Normalize to 0-100
    def norm(param, val):
        r = OPTIMAL_RANGES[param]
        mid = (r["min"] + r["max"]) / 2
        spread = (r["max"] - r["min"]) / 2
        score = max(0, 100 - abs(val - mid) / max(spread, 0.01) * 50)
        return score

    actual_scores = [norm(p, params[p]) for p in cats]
    optimal_scores = [100] * len(cats)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=actual_scores, theta=cats,
        fill="toself", name="Current",
        line_color=PALETTE["primary"],
        fillcolor="rgba(46,204,113,0.2)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=optimal_scores, theta=cats,
        fill="toself", name="Optimal",
        line_color=PALETTE["info"],
        fillcolor="rgba(52,152,219,0.1)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                           gridcolor="#2a3a5c", tickfont=dict(color=PALETTE["text"])),
            angularaxis=dict(gridcolor="#2a3a5c")
        ),
        title="Soil Parameter Health",
        showlegend=True,
        legend=dict(x=0.85, y=1.1),
        **CHART_LAYOUT
    )
    return fig


def weather_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: temperature range + rainfall bars."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["temperature_max"],
        name="Max Temp", line=dict(color=PALETTE["danger"], width=2),
        mode="lines+markers"
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["temperature_min"],
        name="Min Temp", line=dict(color=PALETTE["info"], width=2, dash="dot"),
        fill="tonexty", fillcolor="rgba(231,76,60,0.1)",
        mode="lines+markers"
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=forecast_df["date"], y=forecast_df["rainfall"],
        name="Rainfall (mm)", marker_color=PALETTE["info"],
        opacity=0.7
    ), secondary_y=True)

    fig.update_layout(
        title="7-Day Weather Forecast",
        xaxis=dict(gridcolor="#2a3a5c"),
        **CHART_LAYOUT
    )
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False,
                     gridcolor="#2a3a5c")
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    return fig


def soil_health_gauge(score: int) -> go.Figure:
    """Gauge chart for overall soil health score."""
    color = (PALETTE["danger"] if score < 50 else
             PALETTE["warning"] if score < 70 else PALETTE["primary"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Soil Health Score", "font": {"color": PALETTE["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PALETTE["text"]},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "#2a3a5c",
            "steps": [
                {"range": [0, 50], "color": "rgba(231,76,60,0.15)"},
                {"range": [50, 70], "color": "rgba(243,156,18,0.15)"},
                {"range": [70, 100], "color": "rgba(46,204,113,0.15)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": score}
        }
    ))
    fig.update_layout(height=280, **CHART_LAYOUT)
    return fig


def action_plan_timeline(plan: list) -> go.Figure:
    """Gantt-style timeline for action plan."""
    if not plan:
        return go.Figure()

    priority_colors = {
        "CRITICAL": PALETTE["danger"],
        "HIGH":     PALETTE["warning"],
        "MEDIUM":   PALETTE["info"],
        "LOW":      PALETTE["primary"]
    }

    fig = go.Figure()
    for i, item in enumerate(plan):
        fig.add_trace(go.Bar(
            y=[f"Day {item.get('day', i+1)}: {item.get('category', '')}"],
            x=[1],
            orientation="h",
            marker_color=priority_colors.get(item.get("priority", "LOW"), PALETTE["primary"]),
            text=item.get("action", "")[:60] + "…" if len(item.get("action", "")) > 60 else item.get("action", ""),
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="white", size=11),
            hovertext=item.get("reasoning", ""),
            name=item.get("priority", "")
        ))

    fig.update_layout(
        title="Action Plan Timeline",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(gridcolor="#2a3a5c"),
        barmode="stack",
        showlegend=False,
        height=max(300, len(plan) * 50),
        **CHART_LAYOUT
    )
    return fig
