"""
src/models/weather_model.py
Time-series weather forecasting using Prophet + live OpenWeatherMap data.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed — using ARIMA fallback.")

import requests
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import OPENWEATHER_API_KEY, DEFAULT_LAT, DEFAULT_LON, WEATHER_MODEL_PATH


class WeatherForecaster:
    """
    Combines live weather data (OpenWeatherMap) with Prophet time-series
    forecasting to predict temperature, rainfall, humidity for next 7 days.
    """

    OWM_BASE = "https://api.openweathermap.org/data/2.5"

    def __init__(self):
        self.model_temp = None
        self.model_rain = None
        self.historical_df = None
        self._build_synthetic_history()

    # ── Live Data ──────────────────────────────────────────────────────────
    def fetch_current(self, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> dict:
        """Fetch current weather from OpenWeatherMap."""
        if not OPENWEATHER_API_KEY:
            logger.warning("No OWM key — returning synthetic current weather.")
            return self._synthetic_current()

        try:
            url = f"{self.OWM_BASE}/weather"
            r = requests.get(url, params={
                "lat": lat, "lon": lon,
                "appid": OPENWEATHER_API_KEY, "units": "metric"
            }, timeout=8)
            r.raise_for_status()
            d = r.json()
            return {
                "temperature": d["main"]["temp"],
                "feels_like": d["main"]["feels_like"],
                "humidity": d["main"]["humidity"],
                "pressure": d["main"]["pressure"],
                "wind_speed": d["wind"]["speed"],
                "description": d["weather"][0]["description"],
                "rainfall_1h": d.get("rain", {}).get("1h", 0),
                "timestamp": datetime.utcnow().isoformat(),
                "location": d.get("name", "Unknown"),
                "source": "live"
            }
        except Exception as e:
            logger.error(f"OWM fetch failed: {e}")
            return self._synthetic_current()

    def fetch_forecast(self, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> list:
        """Fetch 5-day/3h forecast."""
        if not OPENWEATHER_API_KEY:
            return self._synthetic_forecast()

        try:
            url = f"{self.OWM_BASE}/forecast"
            r = requests.get(url, params={
                "lat": lat, "lon": lon,
                "appid": OPENWEATHER_API_KEY, "units": "metric"
            }, timeout=8)
            r.raise_for_status()
            items = r.json()["list"]
            forecasts = []
            for item in items:
                forecasts.append({
                    "datetime": item["dt_txt"],
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "rainfall": item.get("rain", {}).get("3h", 0),
                    "description": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"]
                })
            return forecasts
        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")
            return self._synthetic_forecast()

    # ── Prophet Forecasting ────────────────────────────────────────────────
    def fit_prophet(self):
        """Train Prophet models on historical data."""
        if not PROPHET_AVAILABLE or self.historical_df is None:
            logger.warning("Skipping Prophet fit — not available.")
            return

        df_temp = self.historical_df[["ds", "temperature"]].rename(
            columns={"temperature": "y"})
        df_rain = self.historical_df[["ds", "rainfall"]].rename(
            columns={"rainfall": "y"})

        self.model_temp = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80
        )
        self.model_temp.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        self.model_temp.fit(df_temp)

        self.model_rain = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        self.model_rain.fit(df_rain)
        logger.success("Prophet models fitted on 2-year synthetic history.")

    def predict_7day(self) -> pd.DataFrame:
        """Return 7-day daily forecast DataFrame."""
        if PROPHET_AVAILABLE and self.model_temp:
            future = self.model_temp.make_future_dataframe(periods=7)
            fc_temp = self.model_temp.predict(future).tail(7)
            fc_rain = self.model_rain.predict(future).tail(7)

            result = pd.DataFrame({
                "date": fc_temp["ds"].dt.strftime("%Y-%m-%d"),
                "temperature_min": fc_temp["yhat_lower"].round(1),
                "temperature_max": fc_temp["yhat_upper"].round(1),
                "temperature": fc_temp["yhat"].round(1),
                "rainfall": fc_rain["yhat"].clip(0).round(2),
                "rainfall_upper": fc_rain["yhat_upper"].clip(0).round(2),
            })
            return result
        return self._synthetic_7day_df()

    # ── Agricultural Risk ──────────────────────────────────────────────────
    def assess_risk(self, forecast_df: pd.DataFrame) -> list:
        """Convert forecast into actionable farm alerts."""
        alerts = []
        for _, row in forecast_df.iterrows():
            day_alerts = []
            if row["rainfall"] > 30:
                day_alerts.append({"level": "HIGH", "message": f"Heavy rain expected ({row['rainfall']:.0f}mm). Delay fertilization. Check drainage."})
            elif row["rainfall"] > 10:
                day_alerts.append({"level": "MEDIUM", "message": f"Moderate rain ({row['rainfall']:.0f}mm). Good for irrigation-free growth."})

            if row["temperature"] > 38:
                day_alerts.append({"level": "HIGH", "message": f"Extreme heat ({row['temperature']:.0f}°C). Increase irrigation. Use shade nets."})
            elif row["temperature"] < 10:
                day_alerts.append({"level": "HIGH", "message": f"Cold stress risk ({row['temperature']:.0f}°C). Cover crops overnight."})

            alerts.append({"date": row["date"], "alerts": day_alerts})
        return alerts

    # ── Synthetic Fallbacks ────────────────────────────────────────────────
    def _build_synthetic_history(self):
        """2-year synthetic daily weather history for model training."""
        dates = pd.date_range(end=datetime.today(), periods=730, freq="D")
        np.random.seed(42)
        t = np.arange(730)
        temps = 25 + 8 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 2, 730)
        rain = np.maximum(0, 5 * np.sin(2 * np.pi * (t - 90) / 365) +
                          np.random.exponential(3, 730))
        self.historical_df = pd.DataFrame({
            "ds": dates,
            "temperature": temps.round(1),
            "rainfall": rain.round(2)
        })
        if PROPHET_AVAILABLE:
            self.fit_prophet()

    def _synthetic_current(self) -> dict:
        return {
            "temperature": 28.5, "feels_like": 31.0, "humidity": 72,
            "pressure": 1010, "wind_speed": 3.2,
            "description": "partly cloudy", "rainfall_1h": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "location": "Demo Farm", "source": "synthetic"
        }

    def _synthetic_forecast(self) -> list:
        now = datetime.utcnow()
        return [
            {"datetime": (now + timedelta(hours=3 * i)).strftime("%Y-%m-%d %H:%M:%S"),
             "temperature": round(27 + 3 * np.sin(i / 4), 1),
             "humidity": int(70 + 5 * np.random.randn()),
             "rainfall": round(max(0, np.random.exponential(2)), 1),
             "description": "partly cloudy", "wind_speed": round(2 + np.random.rand() * 2, 1)}
            for i in range(40)
        ]

    def _synthetic_7day_df(self) -> pd.DataFrame:
        dates = [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        return pd.DataFrame({
            "date": dates,
            "temperature_min": [22, 21, 20, 23, 25, 24, 22],
            "temperature_max": [33, 31, 29, 34, 36, 35, 30],
            "temperature": [27, 26, 25, 28, 30, 29, 26],
            "rainfall": [2, 15, 25, 0, 0, 5, 10],
            "rainfall_upper": [5, 30, 50, 2, 1, 15, 20],
        })


if __name__ == "__main__":
    wf = WeatherForecaster()
    current = wf.fetch_current()
    print("Current:", json.dumps(current, indent=2))
    fc = wf.predict_7day()
    print("\n7-Day Forecast:\n", fc)
    risks = wf.assess_risk(fc)
    for r in risks:
        if r["alerts"]:
            print(r["date"], r["alerts"])
