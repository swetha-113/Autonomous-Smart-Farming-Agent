"""
tests/test_all.py
Unit tests for Smart Farming Agent components.
Run: pytest tests/ -v
"""
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest


# ── Disease Detector ───────────────────────────────────────────────────────
class TestCropDiseaseDetector:
    def setup_method(self):
        from src.models.cnn_disease_model import CropDiseaseDetector
        self.detector = CropDiseaseDetector()

    def test_predict_returns_dict(self):
        dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.detector.predict(dummy)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self):
        dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.detector.predict(dummy)
        assert "top_prediction" in result
        assert "confidence" in result
        assert "is_healthy" in result
        assert "top3" in result

    def test_confidence_between_0_and_1(self):
        dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.detector.predict(dummy)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top3_length(self):
        dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.detector.predict(dummy)
        assert len(result["top3"]) == 3

    def test_is_healthy_is_bool(self):
        dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = self.detector.predict(dummy)
        assert isinstance(result["is_healthy"], bool)


# ── Soil Analyzer ──────────────────────────────────────────────────────────
class TestSoilHealthAnalyzer:
    def setup_method(self):
        from src.models.soil_model import SoilHealthAnalyzer
        self.analyzer = SoilHealthAnalyzer()
        self.params = {
            "N": 60, "P": 40, "K": 160, "pH": 6.5,
            "moisture": 55, "EC": 0.4, "organic_matter": 3.0,
            "rainfall": 80, "temperature": 27
        }

    def test_analyze_health_returns_score(self):
        result = self.analyzer.analyze_health(self.params)
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 100

    def test_analyze_health_grade(self):
        result = self.analyzer.analyze_health(self.params)
        assert "grade" in result
        assert result["grade"] is not None

    def test_predict_crop_returns_string(self):
        result = self.analyzer.predict_crop(self.params)
        assert "recommended_crop" in result
        assert isinstance(result["recommended_crop"], str)

    def test_low_nitrogen_detected(self):
        p = dict(self.params, N=5)
        result = self.analyzer.analyze_health(p)
        assert any("N" in issue for issue in result.get("issues", []))

    def test_high_ph_detected(self):
        p = dict(self.params, pH=9.0)
        result = self.analyzer.analyze_health(p)
        assert any("pH" in issue for issue in result.get("issues", []))


# ── Weather Forecaster ─────────────────────────────────────────────────────
class TestWeatherForecaster:
    def setup_method(self):
        from src.models.weather_model import WeatherForecaster
        self.forecaster = WeatherForecaster()

    def test_fetch_current_returns_dict(self):
        result = self.forecaster.fetch_current()
        assert isinstance(result, dict)
        assert "temperature" in result
        assert "humidity" in result

    def test_predict_7day_returns_df(self):
        import pandas as pd
        df = self.forecaster.predict_7day()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7

    def test_forecast_has_required_columns(self):
        df = self.forecaster.predict_7day()
        for col in ["date", "temperature", "rainfall"]:
            assert col in df.columns

    def test_assess_risk_returns_list(self):
        df = self.forecaster.predict_7day()
        risks = self.forecaster.assess_risk(df)
        assert isinstance(risks, list)
        assert len(risks) == 7


# ── Data Utils ─────────────────────────────────────────────────────────────
class TestDataUtils:
    def test_validate_valid_params(self):
        from src.utils.data_utils import validate_soil_params
        params = {"N": 60, "P": 40, "K": 150, "pH": 6.5, "moisture": 50}
        valid, errors = validate_soil_params(params)
        assert valid
        assert len(errors) == 0

    def test_validate_invalid_ph(self):
        from src.utils.data_utils import validate_soil_params
        params = {"pH": 15.0}
        valid, errors = validate_soil_params(params)
        assert not valid
        assert len(errors) > 0

    def test_load_image_from_array(self):
        from src.utils.data_utils import load_image
        dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = load_image(dummy)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    def test_dummy_soil_params(self):
        from src.utils.data_utils import dummy_soil_params
        params = dummy_soil_params()
        assert isinstance(params, dict)
        assert "N" in params
        assert "pH" in params


# ── Agent Integration ──────────────────────────────────────────────────────
class TestSmartFarmingAgent:
    def setup_method(self):
        from src.agents.farming_agent import SmartFarmingAgent
        self.agent = SmartFarmingAgent()
        self.soil = {
            "N": 55, "P": 38, "K": 170, "pH": 6.4,
            "moisture": 52, "EC": 0.42, "organic_matter": 3.0,
            "rainfall": 78, "temperature": 27
        }

    def test_analyze_returns_report(self):
        dummy_img = np.random.randint(0, 200, (64, 64, 3), dtype=np.uint8)
        report = self.agent.analyze(image_array=dummy_img, soil_params=self.soil)
        assert isinstance(report, dict)
        assert "disease" in report
        assert "soil" in report
        assert "weather" in report

    def test_report_has_alert_level(self):
        report = self.agent.analyze(soil_params=self.soil)
        assert "alert_level" in report
        assert report["alert_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_action_plan_generated(self):
        report = self.agent.analyze(soil_params=self.soil)
        assert "action_plan" in report
        assert isinstance(report["action_plan"], list)

    def test_quick_weather(self):
        result = self.agent.quick_weather()
        assert "temperature" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
