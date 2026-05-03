"""
config/settings.py - Central configuration for Smart Farming Agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ─── Model Settings ──────────────────────────────────────────────────────────
CNN_MODEL_PATH = "models/saved/crop_disease_cnn.pt"
FORECAST_MODEL_PATH = "models/saved/weather_forecast.pkl"
IMAGE_SIZE = (224, 224)
NUM_DISEASE_CLASSES = 38  # PlantVillage dataset classes

# ─── Disease Classes (PlantVillage) ──────────────────────────────────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# ─── Soil Parameter Ranges ───────────────────────────────────────────────────
SOIL_PARAMS = {
    "nitrogen": {"min": 0, "max": 140, "unit": "kg/ha", "optimal_range": (40, 80)},
    "phosphorus": {"min": 0, "max": 145, "unit": "kg/ha", "optimal_range": (30, 60)},
    "potassium": {"min": 0, "max": 205, "unit": "kg/ha", "optimal_range": (40, 80)},
    "ph": {"min": 3.5, "max": 9.0, "unit": "", "optimal_range": (6.0, 7.5)},
    "moisture": {"min": 0, "max": 100, "unit": "%", "optimal_range": (40, 70)},
    "temperature": {"min": 0, "max": 50, "unit": "°C", "optimal_range": (20, 35)},
    "humidity": {"min": 0, "max": 100, "unit": "%", "optimal_range": (50, 80)},
    "rainfall": {"min": 0, "max": 300, "unit": "mm", "optimal_range": (50, 150)},
}

# ─── Crop Recommendations ─────────────────────────────────────────────────────
CROPS = ["rice", "wheat", "maize", "tomato", "potato", "cotton", "sugarcane", "soybean"]

# ─── Agent Settings ───────────────────────────────────────────────────────────
AGENT_MODEL = "claude-opus-4-5"
MAX_ITERATIONS = 5
CONFIDENCE_THRESHOLD = 0.75

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "🌾 Autonomous Smart Farming Agent"
REFRESH_INTERVAL = 300  # seconds
