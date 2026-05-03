"""
src/agents/farming_agent.py
Autonomous Smart Farming Agent — orchestrates all models and generates
AI-powered recommendations via Claude API.
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import Optional

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ANTHROPIC_API_KEY, DISEASE_REMEDIES
from src.models.cnn_disease_model import CropDiseaseDetector
from src.models.soil_model import SoilHealthAnalyzer
from src.models.weather_model import WeatherForecaster


class SmartFarmingAgent:
    """
    Autonomous agent that:
    1. Detects crop disease from image (CNN)
    2. Analyzes soil health (RF/GB ensemble)
    3. Forecasts weather (Prophet time-series)
    4. Synthesizes all data → actionable recommendations (Claude LLM)
    5. Generates autonomous action plan
    """

    def __init__(self):
        logger.info("Initializing Smart Farming Agent…")
        self.disease_detector = CropDiseaseDetector()
        self.soil_analyzer = SoilHealthAnalyzer()
        self.weather_forecaster = WeatherForecaster()

        self.claude_client = None
        if CLAUDE_AVAILABLE and ANTHROPIC_API_KEY:
            self.claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Claude AI client connected.")
        else:
            logger.warning("Claude API unavailable — using rule-based recommendations.")

        logger.success("Smart Farming Agent ready ✓")

    # ── Main Analysis Pipeline ─────────────────────────────────────────────
    def analyze(
        self,
        image_array: Optional[np.ndarray] = None,
        soil_params: Optional[dict] = None,
        lat: float = None,
        lon: float = None,
        crop_name: str = "Unknown"
    ) -> dict:
        """
        Full analysis pipeline. Returns a comprehensive report dict.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "crop": crop_name,
            "disease": None,
            "soil": None,
            "weather": None,
            "forecast": None,
            "risks": [],
            "recommendations": [],
            "action_plan": [],
            "alert_level": "LOW"
        }

        # 1. Disease detection
        if image_array is not None:
            logger.info("Running disease detection…")
            disease_result = self.disease_detector.predict(image_array)
            report["disease"] = disease_result
            if not disease_result["is_healthy"]:
                remedy = DISEASE_REMEDIES.get(disease_result["top_prediction"], "Consult local agronomist.")
                report["recommendations"].append({
                    "category": "Disease Control",
                    "priority": "HIGH",
                    "action": remedy,
                    "disease": disease_result["top_prediction"],
                    "confidence": disease_result["confidence"]
                })

        # 2. Soil analysis
        if soil_params:
            logger.info("Analyzing soil health…")
            soil_health = self.soil_analyzer.analyze_health(soil_params)
            crop_rec = self.soil_analyzer.predict_crop(soil_params)
            report["soil"] = {"health": soil_health, "crop_recommendation": crop_rec}

            for rec in soil_health.get("recommendations", []):
                if rec:
                    report["recommendations"].append({
                        "category": "Soil Management",
                        "priority": "MEDIUM",
                        "action": rec
                    })

        # 3. Weather
        logger.info("Fetching weather data…")
        kw = {}
        if lat: kw["lat"] = lat
        if lon: kw["lon"] = lon
        current_weather = self.weather_forecaster.fetch_current(**kw)
        report["weather"] = current_weather

        forecast_df = self.weather_forecaster.predict_7day()
        report["forecast"] = forecast_df.to_dict(orient="records")

        weather_risks = self.weather_forecaster.assess_risk(forecast_df)
        for day in weather_risks:
            for alert in day["alerts"]:
                report["risks"].append({
                    "date": day["date"],
                    "level": alert["level"],
                    "message": alert["message"]
                })
                if alert["level"] == "HIGH":
                    report["recommendations"].append({
                        "category": "Weather Adaptation",
                        "priority": "HIGH",
                        "action": alert["message"]
                    })

        # 4. Determine overall alert level
        high_count = sum(1 for r in report["risks"] if r["level"] == "HIGH")
        if high_count >= 3:
            report["alert_level"] = "CRITICAL"
        elif high_count >= 1:
            report["alert_level"] = "HIGH"
        elif report["recommendations"]:
            report["alert_level"] = "MEDIUM"

        # 5. Generate AI action plan
        logger.info("Generating AI recommendations…")
        report["action_plan"] = self._generate_action_plan(report)

        return report

    # ── AI Recommendations ─────────────────────────────────────────────────
    def _generate_action_plan(self, report: dict) -> list:
        """Use Claude to synthesize all findings into a prioritized plan."""
        if self.claude_client:
            return self._claude_recommendations(report)
        return self._rule_based_plan(report)

    def _claude_recommendations(self, report: dict) -> list:
        """Call Claude API for expert agronomic recommendations."""
        prompt = self._build_prompt(report)
        try:
            message = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            text = message.content[0].text
            return self._parse_action_plan(text)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._rule_based_plan(report)

    def _build_prompt(self, report: dict) -> str:
        disease_info = "No image provided."
        if report["disease"]:
            d = report["disease"]
            disease_info = (
                f"Disease detected: {d['top_prediction']} "
                f"(confidence: {d['confidence']:.1%}). "
                f"Plant is {'healthy' if d['is_healthy'] else 'DISEASED'}."
            )

        soil_info = "No soil data provided."
        if report["soil"]:
            h = report["soil"]["health"]
            soil_info = (
                f"Soil health score: {h['overall_score']}/100 ({h['grade']}). "
                f"Issues: {', '.join(h['issues']) if h['issues'] else 'None'}."
            )

        weather_info = f"Current: {report['weather']['temperature']}°C, " \
                       f"humidity {report['weather']['humidity']}%, " \
                       f"{report['weather']['description']}."

        high_risks = [r['message'] for r in report['risks'] if r['level'] == 'HIGH']

        return f"""You are an expert agronomist AI. Analyze the following farm data and provide a prioritized 7-day action plan.

CROP: {report['crop']}
DISEASE ANALYSIS: {disease_info}
SOIL ANALYSIS: {soil_info}
WEATHER: {weather_info}
HIGH-PRIORITY RISKS: {'; '.join(high_risks) if high_risks else 'None'}

Respond with a JSON array of action items. Each item must have:
- "day": day number (1-7)
- "priority": "CRITICAL", "HIGH", "MEDIUM", or "LOW"
- "category": category name (e.g., "Irrigation", "Pest Control", "Fertilization")
- "action": specific actionable instruction
- "reasoning": brief agronomic explanation

Return ONLY the JSON array, no other text."""

    def _parse_action_plan(self, text: str) -> list:
        """Parse Claude's JSON response."""
        try:
            clean = text.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            logger.warning("Could not parse Claude response as JSON.")
            return self._rule_based_plan({})

    def _rule_based_plan(self, report: dict) -> list:
        """Fallback rule-based action plan."""
        plan = []
        day = 1

        disease = report.get("disease")
        if disease and not disease.get("is_healthy", True):
            plan.append({
                "day": day, "priority": "CRITICAL",
                "category": "Disease Control",
                "action": DISEASE_REMEDIES.get(
                    disease["top_prediction"],
                    "Apply broad-spectrum fungicide. Isolate affected plants."
                ),
                "reasoning": f"Disease {disease['top_prediction']} detected with {disease['confidence']:.0%} confidence."
            })
            day += 1

        soil = report.get("soil")
        if soil:
            for rec in soil["health"].get("recommendations", [])[:3]:
                if rec:
                    plan.append({
                        "day": day, "priority": "HIGH",
                        "category": "Soil Amendment",
                        "action": rec,
                        "reasoning": "Based on soil parameter analysis."
                    })
                    day += 1

        for risk in report.get("risks", [])[:3]:
            plan.append({
                "day": day, "priority": risk["level"],
                "category": "Weather Response",
                "action": risk["message"],
                "reasoning": f"Weather forecast alert for {risk['date']}."
            })
            day += 1

        # Always add monitoring
        plan.append({
            "day": 7, "priority": "LOW",
            "category": "Monitoring",
            "action": "Photo-document crop progress. Re-run soil and disease analysis.",
            "reasoning": "Weekly monitoring is essential for autonomous farm management."
        })
        return plan

    # ── Quick Methods ──────────────────────────────────────────────────────
    def quick_disease_check(self, image_array: np.ndarray) -> dict:
        return self.disease_detector.predict(image_array)

    def quick_soil_check(self, params: dict) -> dict:
        return {
            "health": self.soil_analyzer.analyze_health(params),
            "crop_recommendation": self.soil_analyzer.predict_crop(params)
        }

    def quick_weather(self, lat=None, lon=None) -> dict:
        kw = {}
        if lat: kw["lat"] = lat
        if lon: kw["lon"] = lon
        return self.weather_forecaster.fetch_current(**kw)


if __name__ == "__main__":
    agent = SmartFarmingAgent()
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    soil = {"N": 45, "P": 35, "K": 180, "pH": 6.2,
            "moisture": 55, "EC": 0.4, "organic_matter": 3.2,
            "rainfall": 80, "temperature": 28}
    report = agent.analyze(image_array=dummy_img, soil_params=soil, crop_name="Tomato")
    print(json.dumps(report, indent=2, default=str))
