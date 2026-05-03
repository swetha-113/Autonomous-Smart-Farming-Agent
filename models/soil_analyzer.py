"""
models/soil_analyzer.py - Soil quality analysis and crop recommendation engine
Uses XGBoost + rule-based system for soil health scoring and recommendations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from loguru import logger

from config.settings import SOIL_PARAMS, CROPS


class SoilAnalyzer:
    """Analyzes soil parameters and recommends crops & fertilizers."""

    def __init__(self):
        self.crop_model = None
        self.health_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "models/saved/soil_analyzer.pkl"

        if Path(self.model_path).exists():
            self._load()

    def _load(self):
        try:
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            self.crop_model = saved["crop_model"]
            self.health_model = saved["health_model"]
            self.scaler = saved["scaler"]
            self.is_trained = True
        except Exception as e:
            logger.error(f"Failed to load soil models: {e}")

    def _save(self):
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "crop_model": self.crop_model,
                "health_model": self.health_model,
                "scaler": self.scaler
            }, f)

    def train(self, df: pd.DataFrame = None):
        """Train on provided data or generate synthetic training data."""
        if df is None:
            df = self._generate_training_data()

        features = ['nitrogen', 'phosphorus', 'potassium', 'ph',
                    'moisture', 'temperature', 'humidity', 'rainfall']
        X = df[features].values
        y_crop = df['crop'].values
        y_health = df['health_score'].values

        X_scaled = self.scaler.fit_transform(X)

        # Crop recommendation classifier
        self.crop_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        self.crop_model.fit(X_scaled, y_crop)

        # Soil health regressor (0-100 score)
        self.health_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42
        )
        self.health_model.fit(X_scaled, y_health)

        self.is_trained = True
        self._save()
        logger.info("Soil analysis models trained and saved.")

    def _generate_training_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic training data based on agronomic knowledge."""
        np.random.seed(42)
        rows = []

        crop_conditions = {
            "rice": {"N": (80, 120), "P": (40, 60), "K": (40, 60), "pH": (5.5, 7.0), "temp": (20, 35), "rain": (150, 300)},
            "wheat": {"N": (60, 100), "P": (30, 50), "K": (30, 50), "pH": (6.0, 7.5), "temp": (10, 25), "rain": (50, 150)},
            "maize": {"N": (80, 120), "P": (50, 80), "K": (50, 80), "pH": (5.8, 7.0), "temp": (18, 32), "rain": (50, 200)},
            "tomato": {"N": (60, 100), "P": (60, 100), "K": (80, 120), "pH": (6.0, 7.0), "temp": (20, 30), "rain": (40, 100)},
            "potato": {"N": (80, 120), "P": (60, 100), "K": (100, 140), "pH": (5.0, 6.5), "temp": (15, 25), "rain": (50, 150)},
            "cotton": {"N": (70, 110), "P": (40, 70), "K": (50, 90), "pH": (5.8, 8.0), "temp": (21, 35), "rain": (50, 150)},
            "sugarcane": {"N": (100, 140), "P": (30, 50), "K": (40, 60), "pH": (6.0, 7.5), "temp": (24, 38), "rain": (100, 200)},
            "soybean": {"N": (0, 20), "P": (60, 100), "K": (80, 120), "pH": (6.0, 7.0), "temp": (20, 30), "rain": (60, 150)},
        }

        for crop, conds in crop_conditions.items():
            n = n_samples // len(crop_conditions)
            for _ in range(n):
                N = np.random.uniform(*conds["N"])
                P = np.random.uniform(*conds["P"])
                K = np.random.uniform(*conds["K"])
                pH = np.random.uniform(*conds["pH"])
                temp = np.random.uniform(*conds["temp"])
                rain = np.random.uniform(*conds["rain"])
                moisture = np.random.uniform(30, 80)
                humidity = np.random.uniform(40, 90)

                # Health score based on deviation from optimal
                health = 100
                health -= abs(N - np.mean(conds["N"])) * 0.3
                health -= abs(P - np.mean(conds["P"])) * 0.3
                health -= abs(pH - np.mean(conds["pH"])) * 5
                health = max(20, min(100, health + np.random.normal(0, 5)))

                rows.append({
                    "nitrogen": N, "phosphorus": P, "potassium": K,
                    "ph": pH, "moisture": moisture, "temperature": temp,
                    "humidity": humidity, "rainfall": rain,
                    "crop": crop, "health_score": health
                })

        return pd.DataFrame(rows)

    def analyze(self, soil_data: dict) -> dict:
        """
        Full soil analysis with recommendations.
        soil_data: dict with nitrogen, phosphorus, potassium, ph, moisture, temperature, humidity, rainfall
        """
        # Parameter validation
        validation = self._validate_params(soil_data)

        # Health score
        health_score = self._compute_health_score(soil_data)

        # ML-based crop recommendation (fallback to rule-based)
        crop_recommendation = self._recommend_crop(soil_data)

        # Fertilizer recommendations
        fertilizer = self._recommend_fertilizer(soil_data)

        # Irrigation recommendation
        irrigation = self._recommend_irrigation(soil_data)

        # Alerts
        alerts = self._generate_alerts(soil_data, validation)

        return {
            "health_score": health_score,
            "health_grade": self._grade(health_score),
            "crop_recommendation": crop_recommendation,
            "fertilizer_recommendation": fertilizer,
            "irrigation_recommendation": irrigation,
            "parameter_validation": validation,
            "alerts": alerts,
            "summary": self._generate_summary(health_score, crop_recommendation, alerts)
        }

    def _validate_params(self, data: dict) -> dict:
        validation = {}
        for param, config in SOIL_PARAMS.items():
            val = data.get(param)
            if val is None:
                validation[param] = {"status": "missing", "value": None}
                continue
            low, high = config["optimal_range"]
            if val < low:
                status = "low"
            elif val > high:
                status = "high"
            else:
                status = "optimal"
            validation[param] = {
                "status": status,
                "value": val,
                "optimal_range": config["optimal_range"],
                "unit": config["unit"]
            }
        return validation

    def _compute_health_score(self, data: dict) -> float:
        if self.is_trained:
            try:
                features = np.array([[
                    data.get("nitrogen", 60), data.get("phosphorus", 40),
                    data.get("potassium", 40), data.get("ph", 6.5),
                    data.get("moisture", 50), data.get("temperature", 25),
                    data.get("humidity", 60), data.get("rainfall", 80)
                ]])
                features_scaled = self.scaler.transform(features)
                score = self.health_model.predict(features_scaled)[0]
                return round(float(np.clip(score, 0, 100)), 1)
            except Exception:
                pass

        # Rule-based fallback
        score = 100.0
        penalties = {
            "nitrogen": (40, 80, 15),
            "phosphorus": (30, 60, 12),
            "potassium": (40, 80, 12),
            "ph": (6.0, 7.5, 20),
            "moisture": (40, 70, 10),
        }
        for param, (low, high, weight) in penalties.items():
            val = data.get(param, (low + high) / 2)
            if val < low:
                score -= weight * (low - val) / low
            elif val > high:
                score -= weight * (val - high) / high

        return round(max(0, min(100, score)), 1)

    def _recommend_crop(self, data: dict) -> list:
        if self.is_trained:
            try:
                features = np.array([[
                    data.get("nitrogen", 60), data.get("phosphorus", 40),
                    data.get("potassium", 40), data.get("ph", 6.5),
                    data.get("moisture", 50), data.get("temperature", 25),
                    data.get("humidity", 60), data.get("rainfall", 80)
                ]])
                features_scaled = self.scaler.transform(features)
                proba = self.crop_model.predict_proba(features_scaled)[0]
                classes = self.crop_model.classes_
                top3_idx = np.argsort(proba)[-3:][::-1]
                return [
                    {"crop": classes[i], "confidence": round(float(proba[i]) * 100, 1)}
                    for i in top3_idx
                ]
            except Exception:
                pass

        # Rule-based fallback
        ph = data.get("ph", 6.5)
        temp = data.get("temperature", 25)
        rain = data.get("rainfall", 80)

        if rain > 150 and temp > 22:
            return [{"crop": "rice", "confidence": 82}, {"crop": "sugarcane", "confidence": 68}]
        elif temp < 20:
            return [{"crop": "wheat", "confidence": 78}, {"crop": "potato", "confidence": 65}]
        else:
            return [{"crop": "maize", "confidence": 74}, {"crop": "tomato", "confidence": 61}]

    def _recommend_fertilizer(self, data: dict) -> dict:
        N = data.get("nitrogen", 60)
        P = data.get("phosphorus", 40)
        K = data.get("potassium", 40)
        ph = data.get("ph", 6.5)

        recommendations = {}
        if N < 40:
            recommendations["nitrogen"] = {"type": "Urea (46-0-0)", "amount": f"{int((60-N)*2)} kg/ha", "priority": "high"}
        if P < 30:
            recommendations["phosphorus"] = {"type": "DAP (18-46-0)", "amount": f"{int((50-P)*1.5)} kg/ha", "priority": "high"}
        if K < 40:
            recommendations["potassium"] = {"type": "MOP (0-0-60)", "amount": f"{int((60-K)*1.2)} kg/ha", "priority": "medium"}
        if ph < 6.0:
            recommendations["ph_correction"] = {"type": "Agricultural Lime", "amount": f"{int((6.0-ph)*500)} kg/ha", "priority": "high"}
        elif ph > 7.5:
            recommendations["ph_correction"] = {"type": "Sulfur", "amount": f"{int((ph-7.5)*200)} kg/ha", "priority": "medium"}

        if not recommendations:
            recommendations["status"] = "Soil nutrients are within optimal range. Maintain with balanced fertilization."

        return recommendations

    def _recommend_irrigation(self, data: dict) -> dict:
        moisture = data.get("moisture", 50)
        temperature = data.get("temperature", 25)
        rainfall = data.get("rainfall", 80)

        if moisture < 30:
            frequency = "Daily"
            amount = "25-30 mm"
            urgency = "High"
        elif moisture < 50:
            frequency = "Every 2-3 days"
            amount = "15-20 mm"
            urgency = "Medium"
        else:
            frequency = "Weekly or as needed"
            amount = "10-15 mm"
            urgency = "Low"

        return {
            "frequency": frequency,
            "amount_per_session": amount,
            "urgency": urgency,
            "method": "Drip irrigation recommended" if temperature > 30 else "Sprinkler or flood irrigation",
            "note": "Reduce irrigation if rainfall > 20mm expected in next 48 hours"
        }

    def _generate_alerts(self, data: dict, validation: dict) -> list:
        alerts = []
        for param, info in validation.items():
            if info.get("status") in ["low", "high"]:
                severity = "critical" if param == "ph" else "warning"
                alerts.append({
                    "type": severity,
                    "parameter": param,
                    "message": f"{param.capitalize()} is {info['status']} ({info['value']} {info.get('unit','')})",
                    "action": f"Adjust {param} to optimal range {info.get('optimal_range', 'N/A')}"
                })
        return alerts

    def _grade(self, score: float) -> str:
        if score >= 85: return "Excellent"
        if score >= 70: return "Good"
        if score >= 55: return "Fair"
        if score >= 40: return "Poor"
        return "Critical"

    def _generate_summary(self, health_score: float, crops: list, alerts: list) -> str:
        top_crop = crops[0]["crop"] if crops and isinstance(crops[0], dict) else "mixed crops"
        grade = self._grade(health_score)
        alert_count = len(alerts)
        return (
            f"Soil health is {grade} ({health_score}/100). "
            f"Best suited for {top_crop}. "
            f"{'No immediate issues detected.' if alert_count == 0 else f'{alert_count} issue(s) require attention.'}"
        )
