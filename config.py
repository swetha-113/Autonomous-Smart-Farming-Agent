"""
config.py — Central configuration for Smart Farming Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ── Location ───────────────────────────────────────────────────────────────
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", 12.9716))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", 77.5946))
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "Bangalore, India")

# ── Model Paths ────────────────────────────────────────────────────────────
CNN_MODEL_PATH = MODELS_DIR / "crop_disease_cnn.h5"
SOIL_MODEL_PATH = MODELS_DIR / "soil_health_model.pkl"
WEATHER_MODEL_PATH = MODELS_DIR / "weather_forecast_model.pkl"

# ── Disease Classes (PlantVillage 38-class) ────────────────────────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
    "Cherry___healthy", "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

DISEASE_REMEDIES = {
    "Apple___Apple_scab": "Apply fungicide (captan/mancozeb). Remove fallen leaves. Prune for airflow.",
    "Apple___Black_rot": "Prune infected branches. Apply copper-based fungicide. Destroy mummified fruits.",
    "Apple___Cedar_apple_rust": "Apply myclobutanil fungicide early spring. Remove nearby juniper hosts.",
    "Corn___Common_rust": "Apply triazole fungicide. Plant resistant varieties. Ensure adequate spacing.",
    "Corn___Northern_Leaf_Blight": "Apply strobilurin fungicide. Rotate crops. Use resistant hybrids.",
    "Potato___Early_blight": "Apply chlorothalonil. Remove infected foliage. Ensure proper irrigation.",
    "Potato___Late_blight": "Apply mefenoxam-based fungicide immediately. Destroy infected plants.",
    "Tomato___Bacterial_spot": "Apply copper bactericide. Remove infected leaves. Avoid overhead watering.",
    "Tomato___Early_blight": "Apply mancozeb fungicide. Mulch base. Stake plants for airflow.",
    "Tomato___Late_blight": "Apply copper fungicide. Destroy infected plants. Avoid wet foliage.",
    "Tomato___Leaf_Mold": "Improve ventilation. Apply fungicide. Reduce humidity in greenhouse.",
    "Tomato___Septoria_leaf_spot": "Remove lower leaves. Apply fungicide. Avoid wetting foliage.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Control whitefly vectors. Remove infected plants. Use reflective mulch.",
    "Grape___Black_rot": "Apply myclobutanil pre-bloom. Remove mummified fruit. Prune for airflow.",
}

SOIL_PARAMETERS = ["N", "P", "K", "pH", "moisture", "EC", "organic_matter"]

CROPS = ["Tomato", "Potato", "Corn", "Apple", "Grape", "Pepper", "Strawberry",
         "Peach", "Cherry", "Soybean", "Wheat", "Rice", "Cotton"]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
