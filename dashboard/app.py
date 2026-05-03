"""
dashboard/app.py
Smart Farming Agent — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image

# Path setup
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DISEASE_CLASSES, CROPS, OPTIMAL_RANGES
from src.agents.farming_agent import SmartFarmingAgent
from src.utils.data_utils import load_image, validate_soil_params, dummy_soil_params
from src.utils.visualizations import (
    disease_confidence_chart, soil_radar_chart,
    weather_forecast_chart, soil_health_gauge,
    action_plan_timeline
)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Farming Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #2ECC71;
    --danger:  #E74C3C;
    --warning: #F39C12;
    --info:    #3498DB;
    --dark:    #0D1117;
    --surface: #161B22;
    --card:    #1C2333;
    --border:  #30363D;
    --text:    #E6EDF3;
    --muted:   #8B949E;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text);
}

.stApp { background: var(--dark); }

/* Cards */
.farm-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* Metric tiles */
.metric-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: var(--primary); }
.metric-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

/* Alert badges */
.alert-critical { background: rgba(231,76,60,0.15); border-left: 4px solid #E74C3C; padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
.alert-high     { background: rgba(243,156,18,0.15); border-left: 4px solid #F39C12; padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
.alert-medium   { background: rgba(52,152,219,0.15); border-left: 4px solid #3498DB; padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
.alert-low      { background: rgba(46,204,113,0.15); border-left: 4px solid #2ECC71; padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: var(--primary);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.5rem; margin-bottom: 1rem;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), #27AE60) !important;
    color: #0D1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(46,204,113,0.4); }

/* Progress bars */
.stProgress > div > div { background: var(--primary) !important; }

/* Inputs */
.stSlider, .stSelectbox, .stFileUploader { color: var(--text); }

/* Header */
.farm-header {
    background: linear-gradient(135deg, #0D1117 0%, #1C2333 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.farm-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(46,204,113,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.farm-title { font-size: 2.2rem; font-weight: 700; color: var(--primary); margin: 0; }
.farm-subtitle { color: var(--muted); margin-top: 0.3rem; font-size: 1rem; }
.status-badge {
    display: inline-block;
    background: rgba(46,204,113,0.15);
    border: 1px solid var(--primary);
    color: var(--primary);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading farming agent models…")
def get_agent():
    return SmartFarmingAgent()


if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "report" not in st.session_state:
    st.session_state.report = None


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="farm-header">
    <div class="farm-title">🌾 Smart Farming Agent</div>
    <div class="farm-subtitle">Autonomous AI-Powered Crop Disease Detection & Farm Management</div>
    <span class="status-badge">● SYSTEM ONLINE</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    crop_name = st.selectbox("Select Crop", CROPS, index=0)

    st.markdown("### 📍 Farm Location")
    lat = st.number_input("Latitude", value=12.9716, format="%.4f")
    lon = st.number_input("Longitude", value=77.5946, format="%.4f")

    st.markdown("### 🌱 Soil Parameters")
    use_demo = st.checkbox("Use demo soil data", value=True)

    if use_demo:
        soil_params = dummy_soil_params()
        st.info("Using demo soil values. Uncheck to enter custom data.")
    else:
        soil_params = {}
        col1, col2 = st.columns(2)
        with col1:
            soil_params["N"] = st.slider("N (mg/kg)", 0, 200, 60)
            soil_params["P"] = st.slider("P (mg/kg)", 0, 150, 42)
            soil_params["K"] = st.slider("K (mg/kg)", 0, 400, 165)
            soil_params["pH"] = st.slider("pH", 3.0, 10.0, 6.5, 0.1)
        with col2:
            soil_params["moisture"] = st.slider("Moisture (%)", 0, 100, 52)
            soil_params["EC"] = st.slider("EC (dS/m)", 0.0, 5.0, 0.45, 0.05)
            soil_params["organic_matter"] = st.slider("Organic Matter (%)", 0.0, 15.0, 3.1, 0.1)
            soil_params["rainfall"] = st.slider("Rainfall (mm)", 0, 400, 75)
            soil_params["temperature"] = st.slider("Temp (°C)", -5, 50, 27)

    st.divider()
    st.markdown("### 🖼️ Crop Image")
    uploaded_file = st.file_uploader(
        "Upload crop/leaf image", type=["jpg", "jpeg", "png", "webp"]
    )
    use_demo_img = st.checkbox("Use demo image (random)", value=True)

    st.divider()
    analyze_btn = st.button("🔬 Run Full Analysis", use_container_width=True)


# ── Navigation Tabs ────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Dashboard", "🦠 Disease Detection",
                "🌱 Soil Analysis", "☁️ Weather & Forecast", "📋 Action Plan"])


# ── Run Analysis ───────────────────────────────────────────────────────────
if analyze_btn:
    agent = get_agent()
    image_array = None

    if uploaded_file:
        image_array = load_image(uploaded_file)
    elif use_demo_img:
        image_array = np.random.randint(60, 200, (256, 256, 3), dtype=np.uint8)
        image_array[:, :, 1] = np.clip(image_array[:, :, 1] + 40, 0, 255)

    valid, errors = validate_soil_params(soil_params)
    if not valid:
        st.error(f"Soil parameter errors: {', '.join(errors)}")
    else:
        with st.spinner("🤖 Autonomous agent analyzing your farm data…"):
            report = agent.analyze(
                image_array=image_array,
                soil_params=soil_params,
                lat=lat, lon=lon,
                crop_name=crop_name
            )
        st.session_state.report = report
        st.session_state.analysis_done = True
        st.success("✅ Analysis complete!")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Dashboard
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if not st.session_state.analysis_done:
        st.markdown("""
        <div class="farm-card" style="text-align:center; padding: 3rem;">
            <div style="font-size:4rem;">🌾</div>
            <h2 style="color:#2ECC71;">Welcome to Smart Farming Agent</h2>
            <p style="color:#8B949E; max-width:500px; margin:0 auto;">
                Configure your farm settings in the sidebar and click
                <strong style="color:#2ECC71;">Run Full Analysis</strong> to get
                AI-powered crop disease detection, soil health scoring,
                weather forecasting, and a personalized action plan.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        report = st.session_state.report
        alert_colors = {"LOW": "#2ECC71", "MEDIUM": "#3498DB", "HIGH": "#F39C12", "CRITICAL": "#E74C3C"}
        al = report["alert_level"]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            disease = report.get("disease") or {}
            healthy = disease.get("is_healthy", True)
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{"✅" if healthy else "⚠️"}</div>
                <div style="font-size:1rem;color:{'#2ECC71' if healthy else '#E74C3C'};font-weight:600;">
                    {"Healthy" if healthy else "Disease Found"}
                </div>
                <div class="metric-label">Crop Status</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            soil_score = report.get("soil", {}).get("health", {}).get("overall_score", "—")
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{soil_score}</div>
                <div class="metric-label">Soil Health /100</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            temp = report.get("weather", {}).get("temperature", "—")
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value">{temp}°</div>
                <div class="metric-label">Temperature °C</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-tile">
                <div class="metric-value" style="color:{alert_colors[al]};">{al}</div>
                <div class="metric-label">Alert Level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risks & quick recommendations
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="section-header">🚨 Active Alerts</div>', unsafe_allow_html=True)
            risks = report.get("risks", [])
            if risks:
                for r in risks[:6]:
                    level_class = f"alert-{r['level'].lower()}"
                    st.markdown(f'<div class="{level_class}"><strong>{r["level"]}</strong> — {r["message"]}</div>',
                                unsafe_allow_html=True)
            else:
                st.success("No high-priority alerts. Farm conditions look good!")

        with col2:
            st.markdown('<div class="section-header">💡 Top Recommendations</div>', unsafe_allow_html=True)
            for rec in report.get("recommendations", [])[:4]:
                icon = {"Disease Control": "🦠", "Soil Management": "🌱",
                        "Weather Adaptation": "☁️"}.get(rec.get("category"), "📌")
                priority_color = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#2ECC71"}.get(
                    rec.get("priority", "LOW"), "#3498DB")
                st.markdown(f"""<div class="farm-card" style="margin-bottom:0.5rem;">
                    <span style="color:{priority_color};font-weight:600;">{icon} [{rec['priority']}]</span>
                    <br><span style="font-size:0.9rem;">{rec['action'][:120]}{'…' if len(rec.get('action',''))>120 else ''}</span>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Disease Detection
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">🦠 Crop Disease Detection (CNN)</div>', unsafe_allow_html=True)
    if st.session_state.analysis_done:
        report = st.session_state.report
        disease = report.get("disease") or {}

        col1, col2 = st.columns([1, 1.5])
        with col1:
            if "disease" in report and report["disease"]:
                top = disease["top_prediction"]
                conf = disease["confidence"]
                healthy = disease["is_healthy"]

                st.markdown(f"""<div class="farm-card">
                    <div style="font-size:3rem;text-align:center;">{'🌿' if healthy else '🦠'}</div>
                    <h3 style="text-align:center;color:{'#2ECC71' if healthy else '#E74C3C'};">
                        {'Plant Healthy' if healthy else 'Disease Detected'}
                    </h3>
                    <p style="text-align:center;color:#8B949E;font-size:0.85rem;">
                        {top.replace('___', ' — ').replace('_', ' ')}
                    </p>
                    <div style="text-align:center;">
                        <strong style="font-size:1.5rem;color:{'#2ECC71' if healthy else '#E74C3C'};">
                            {conf:.1%}
                        </strong>
                        <div style="color:#8B949E;font-size:0.8rem;">Confidence</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                if not healthy:
                    from config import DISEASE_REMEDIES
                    remedy = DISEASE_REMEDIES.get(top, "Consult a local agronomist for treatment advice.")
                    st.markdown(f"""<div class="alert-high">
                        <strong>🩺 Recommended Treatment</strong><br>
                        <span style="font-size:0.9rem;">{remedy}</span>
                    </div>""", unsafe_allow_html=True)

        with col2:
            if disease.get("top3"):
                st.plotly_chart(
                    disease_confidence_chart(disease["top3"]),
                    use_container_width=True
                )
    else:
        st.info("Run analysis to see disease detection results.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Soil Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🌱 Soil Health Analysis</div>', unsafe_allow_html=True)
    if st.session_state.analysis_done:
        report = st.session_state.report
        soil_data = report.get("soil", {})
        health = soil_data.get("health", {})
        crop_rec = soil_data.get("crop_recommendation", {})

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(soil_health_gauge(health.get("overall_score", 50)),
                            use_container_width=True)

            st.markdown(f"""<div class="farm-card">
                <strong>🌾 Recommended Crop:</strong>
                <span style="color:#2ECC71;font-size:1.1rem;font-weight:700;">
                    {crop_rec.get('recommended_crop', '—')}
                </span>
                <span style="color:#8B949E;font-size:0.85rem;">
                    ({crop_rec.get('confidence', 0):.1%} match)
                </span>
            </div>""", unsafe_allow_html=True)

            for issue in health.get("issues", []):
                st.markdown(f'<div class="alert-medium">⚠️ {issue}</div>', unsafe_allow_html=True)

        with col2:
            st.plotly_chart(
                soil_radar_chart(soil_params, health.get("parameter_scores", {})),
                use_container_width=True
            )

        # Parameter table
        st.markdown('<div class="section-header">Parameter Details</div>', unsafe_allow_html=True)
        from src.utils.data_utils import generate_soil_report_df
        df = generate_soil_report_df(soil_params, health)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Run analysis to see soil health results.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Weather & Forecast
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">☁️ Weather Analysis & Forecast</div>', unsafe_allow_html=True)
    if st.session_state.analysis_done:
        report = st.session_state.report
        weather = report.get("weather", {})
        forecast = report.get("forecast", [])

        # Current weather
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("🌡️", "Temperature", f"{weather.get('temperature', '—')}°C"),
            ("💧", "Humidity", f"{weather.get('humidity', '—')}%"),
            ("💨", "Wind", f"{weather.get('wind_speed', '—')} m/s"),
            ("🌧️", "Rain 1h", f"{weather.get('rainfall_1h', 0)} mm"),
        ]
        for col, (icon, label, val) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""<div class="metric-tile">
                    <div style="font-size:2rem;">{icon}</div>
                    <div class="metric-value" style="font-size:1.5rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"**Conditions:** {weather.get('description', '').title()} | "
                    f"**Source:** {weather.get('source', '—').upper()} | "
                    f"**Location:** {weather.get('location', '—')}")

        if forecast:
            fc_df = pd.DataFrame(forecast)
            st.plotly_chart(weather_forecast_chart(fc_df), use_container_width=True)

        # Weather risks
        st.markdown('<div class="section-header">🌩️ Weather Alerts</div>', unsafe_allow_html=True)
        for day_risk in report.get("risks", []):
            level_class = f"alert-{day_risk['level'].lower()}"
            st.markdown(
                f'<div class="{level_class}"><strong>{day_risk.get("date","")} — {day_risk["level"]}</strong>: {day_risk["message"]}</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("Run analysis to see weather forecast.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Action Plan
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">📋 Autonomous Action Plan</div>', unsafe_allow_html=True)
    if st.session_state.analysis_done:
        report = st.session_state.report
        plan = report.get("action_plan", [])

        if plan:
            st.plotly_chart(action_plan_timeline(plan), use_container_width=True)

            st.markdown("### Detailed Steps")
            priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            sorted_plan = sorted(plan, key=lambda x: priority_order.get(x.get("priority", "LOW"), 3))

            for item in sorted_plan:
                priority = item.get("priority", "LOW")
                colors = {"CRITICAL": "#E74C3C", "HIGH": "#F39C12", "MEDIUM": "#3498DB", "LOW": "#2ECC71"}
                color = colors.get(priority, "#2ECC71")
                with st.expander(f"Day {item.get('day','?')} | {item.get('category','—')} [{priority}]"):
                    st.markdown(f"**Action:** {item.get('action', '—')}")
                    st.markdown(f"*Reasoning: {item.get('reasoning', '—')}*")
        else:
            st.info("No action plan generated. Run analysis first.")
    else:
        st.info("Run analysis to generate your personalized action plan.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#8B949E;font-size:0.8rem;">
    🌾 Smart Farming Agent • AI-Powered by CNN + Prophet + Claude AI<br>
    Built for autonomous precision agriculture
</div>
""", unsafe_allow_html=True)
