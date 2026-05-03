"""
src/models/soil_model.py
ML model for soil health analysis and crop recommendation.
Uses Random Forest + Gradient Boosting ensemble.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import SOIL_MODEL_PATH, CROPS


OPTIMAL_RANGES = {
    "N":             {"min": 40,  "max": 80,  "unit": "mg/kg"},
    "P":             {"min": 20,  "max": 60,  "unit": "mg/kg"},
    "K":             {"min": 100, "max": 200, "unit": "mg/kg"},
    "pH":            {"min": 6.0, "max": 7.5, "unit": ""},
    "moisture":      {"min": 30,  "max": 60,  "unit": "%"},
    "EC":            {"min": 0.2, "max": 0.8, "unit": "dS/m"},
    "organic_matter":{"min": 2.5, "max": 5.0, "unit": "%"},
}


class SoilHealthAnalyzer:
    """
    Analyzes soil parameters and recommends:
    1. Top crop to plant
    2. Soil health score (0–100)
    3. Fertilizer recommendations
    """

    FEATURES = ["N", "P", "K", "pH", "moisture", "EC", "organic_matter",
                "rainfall", "temperature"]

    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded = False
        self._try_load()

    # ── Training ───────────────────────────────────────────────────────────
    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic soil dataset for training."""
        np.random.seed(42)
        data = []
        crop_profiles = {
            "Rice":       {"N":(60,90),"P":(25,50),"K":(30,60),"pH":(5.5,7.0),"moisture":(60,80),"EC":(0.2,0.5),"organic_matter":(2,4),"rainfall":(150,300),"temperature":(22,35)},
            "Wheat":      {"N":(50,80),"P":(30,60),"K":(40,80),"pH":(6.0,7.5),"moisture":(40,65),"EC":(0.2,0.6),"organic_matter":(1.5,3.5),"rainfall":(50,150),"temperature":(15,25)},
            "Tomato":     {"N":(55,90),"P":(40,70),"K":(120,200),"pH":(6.0,7.0),"moisture":(50,70),"EC":(0.3,0.8),"organic_matter":(2,5),"rainfall":(60,120),"temperature":(20,30)},
            "Potato":     {"N":(70,100),"P":(50,80),"K":(150,220),"pH":(5.5,6.5),"moisture":(55,75),"EC":(0.3,0.7),"organic_matter":(2,4),"rainfall":(50,100),"temperature":(15,20)},
            "Corn":       {"N":(80,120),"P":(30,60),"K":(100,180),"pH":(6.0,7.0),"moisture":(50,70),"EC":(0.2,0.5),"organic_matter":(1.5,3),"rainfall":(50,200),"temperature":(20,35)},
            "Cotton":     {"N":(50,80),"P":(30,60),"K":(80,150),"pH":(6.5,8.0),"moisture":(40,65),"EC":(0.2,0.6),"organic_matter":(1,2.5),"rainfall":(60,100),"temperature":(25,35)},
            "Soybean":    {"N":(20,40),"P":(40,80),"K":(80,160),"pH":(6.0,7.0),"moisture":(45,70),"EC":(0.2,0.5),"organic_matter":(2,4),"rainfall":(60,120),"temperature":(20,30)},
            "Sugarcane":  {"N":(100,150),"P":(25,50),"K":(120,200),"pH":(6.0,7.5),"moisture":(65,80),"EC":(0.3,0.7),"organic_matter":(1.5,3),"rainfall":(150,250),"temperature":(25,35)},
        }
        for crop, profile in crop_profiles.items():
            n = n_samples // len(crop_profiles)
            for _ in range(n):
                row = {k: np.random.uniform(*v) for k, v in profile.items()}
                row["crop"] = crop
                data.append(row)
        df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def train(self, df: pd.DataFrame = None):
        if df is None:
            logger.info("Generating synthetic soil dataset…")
            df = self.generate_synthetic_data()

        X = df[self.FEATURES]
        y = df["crop"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=15,
                                           min_samples_leaf=2, random_state=42,
                                           n_jobs=-1))
        ])
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        logger.info(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        logger.info(f"\n{classification_report(y_test, pipeline.predict(X_test))}")

        joblib.dump(pipeline, SOIL_MODEL_PATH)
        self.model = pipeline
        self.loaded = True
        logger.success(f"Soil model saved → {SOIL_MODEL_PATH}")

    # ── Inference ──────────────────────────────────────────────────────────
    def predict_crop(self, soil_params: dict) -> dict:
        """Return recommended crop + probabilities."""
        features = self._build_features(soil_params)
        if self.loaded and self.model:
            probs = self.model.predict_proba([features])[0]
            classes = self.model.classes_
            top_idx = np.argsort(probs)[::-1][:3]
            return {
                "recommended_crop": classes[top_idx[0]],
                "confidence": float(probs[top_idx[0]]),
                "top3": [{"crop": classes[i], "probability": float(probs[i])} for i in top_idx]
            }
        # Fallback heuristic
        return self._heuristic_recommend(soil_params)

    def analyze_health(self, soil_params: dict) -> dict:
        """Score soil health 0-100 and flag issues."""
        scores = {}
        issues = []
        recommendations = []

        for param, val in soil_params.items():
            if param not in OPTIMAL_RANGES:
                continue
            r = OPTIMAL_RANGES[param]
            if r["min"] <= val <= r["max"]:
                scores[param] = 100
            elif val < r["min"]:
                deficit = (r["min"] - val) / r["min"]
                scores[param] = max(0, int((1 - deficit) * 100))
                issues.append(f"Low {param} ({val:.1f} {r['unit']})")
                recommendations.append(self._fertilizer_rec(param, "low", val, r))
            else:
                excess = (val - r["max"]) / r["max"]
                scores[param] = max(0, int((1 - excess * 0.5) * 100))
                issues.append(f"High {param} ({val:.1f} {r['unit']})")
                recommendations.append(self._fertilizer_rec(param, "high", val, r))

        overall = int(np.mean(list(scores.values()))) if scores else 50
        return {
            "overall_score": overall,
            "parameter_scores": scores,
            "issues": issues,
            "recommendations": recommendations,
            "grade": self._grade(overall)
        }

    def _fertilizer_rec(self, param, status, val, r) -> str:
        recs = {
            ("N", "low"):  "Apply urea (46-0-0) at 50–80 kg/ha or compost.",
            ("N", "high"): "Reduce nitrogen inputs. Leach field if possible.",
            ("P", "low"):  "Apply DAP (18-46-0) at 30–50 kg/ha.",
            ("P", "high"): "Avoid phosphorus fertilizers. Use P-fixing crops.",
            ("K", "low"):  "Apply MOP (0-0-60) at 40–60 kg/ha.",
            ("K", "high"): "Reduce potash. Use calcium/magnesium amendments.",
            ("pH", "low"): "Apply agricultural lime at 1–2 t/ha to raise pH.",
            ("pH", "high"):"Apply elemental sulfur at 0.5–1 t/ha to lower pH.",
            ("moisture", "low"): "Increase irrigation frequency. Apply mulch.",
            ("moisture", "high"):"Improve drainage. Reduce irrigation.",
            ("EC", "low"): "Apply balanced fertilizer to increase salinity slightly.",
            ("EC", "high"):"Leach soil with fresh water. Reduce fertilizer input.",
            ("organic_matter", "low"): "Add compost/FYM at 10 t/ha.",
            ("organic_matter", "high"):"Maintain — excellent organic matter level.",
        }
        return recs.get((param, status), f"Adjust {param} toward range {r['min']}–{r['max']} {r['unit']}")

    def _build_features(self, params: dict) -> list:
        return [params.get(f, 0) for f in self.FEATURES]

    def _heuristic_recommend(self, p: dict) -> dict:
        if p.get("moisture", 50) > 60:
            return {"recommended_crop": "Rice", "confidence": 0.6, "top3": []}
        if p.get("pH", 7) < 6.0:
            return {"recommended_crop": "Potato", "confidence": 0.55, "top3": []}
        return {"recommended_crop": "Tomato", "confidence": 0.5, "top3": []}

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 85: return "Excellent 🌟"
        if score >= 70: return "Good ✅"
        if score >= 50: return "Fair ⚠️"
        return "Poor ❌"

    def _try_load(self):
        if SOIL_MODEL_PATH.exists():
            try:
                self.model = joblib.load(SOIL_MODEL_PATH)
                self.loaded = True
                logger.info(f"Soil model loaded from {SOIL_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load soil model: {e}")
        else:
            logger.info("No saved soil model found. Training on synthetic data…")
            self.train()


if __name__ == "__main__":
    analyzer = SoilHealthAnalyzer()
    params = {"N": 45, "P": 35, "K": 180, "pH": 6.2,
              "moisture": 55, "EC": 0.4, "organic_matter": 3.2,
              "rainfall": 80, "temperature": 28}
    print("Crop recommendation:", analyzer.predict_crop(params))
    print("Soil health:", analyzer.analyze_health(params))
