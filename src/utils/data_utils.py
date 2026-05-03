"""
src/utils/data_utils.py
Data loading, preprocessing, and augmentation utilities.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from loguru import logger
from typing import Tuple, Optional
import io


def load_image(source) -> np.ndarray:
    """
    Load image from file path, bytes, or PIL Image.
    Returns uint8 numpy array (H, W, 3).
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.uint8)

    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    elif hasattr(source, "read"):        # file-like (UploadedFile from Streamlit)
        img = Image.open(source).convert("RGB")
    elif isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    return np.array(img)


def preprocess_image(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize and normalise to [0, 1]."""
    pil = Image.fromarray(img).resize(target_size, Image.LANCZOS)
    return np.array(pil).astype(np.float32) / 255.0


def validate_soil_params(params: dict) -> Tuple[bool, list]:
    """Check that soil parameters are within physically plausible ranges."""
    RANGES = {
        "N":              (0, 300),
        "P":              (0, 200),
        "K":              (0, 500),
        "pH":             (3.0, 10.0),
        "moisture":       (0, 100),
        "EC":             (0, 8.0),
        "organic_matter": (0, 20),
        "rainfall":       (0, 500),
        "temperature":    (-10, 55),
    }
    errors = []
    for key, val in params.items():
        if key in RANGES:
            lo, hi = RANGES[key]
            if not (lo <= val <= hi):
                errors.append(f"{key}={val} out of range [{lo}, {hi}]")
    return (len(errors) == 0), errors


def generate_soil_report_df(params: dict, health_result: dict) -> pd.DataFrame:
    """Convert soil analysis results to a display-ready DataFrame."""
    from config import OPTIMAL_RANGES
    rows = []
    for param, val in params.items():
        if param not in OPTIMAL_RANGES:
            continue
        r = OPTIMAL_RANGES[param]
        status = "✅ OK"
        if val < r["min"]:
            status = "⬇ Low"
        elif val > r["max"]:
            status = "⬆ High"
        score = health_result.get("parameter_scores", {}).get(param, "—")
        rows.append({
            "Parameter": param,
            "Value": round(val, 2),
            "Unit": r["unit"],
            "Optimal Min": r["min"],
            "Optimal Max": r["max"],
            "Status": status,
            "Score": score
        })
    return pd.DataFrame(rows)


def parse_forecast_to_df(forecast_list: list) -> pd.DataFrame:
    """Convert forecast list-of-dicts to a clean DataFrame."""
    df = pd.DataFrame(forecast_list)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    return df


def dummy_soil_params() -> dict:
    """Return realistic demo soil parameters."""
    return {
        "N": 58.0, "P": 42.0, "K": 165.0, "pH": 6.5,
        "moisture": 52.0, "EC": 0.45, "organic_matter": 3.1,
        "rainfall": 75.0, "temperature": 27.0
    }


def dummy_image() -> np.ndarray:
    """Return a green-tinted placeholder image for demo."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 1] = 128   # green channel
    img[:, :, 0] = 30
    img[:, :, 2] = 20
    return img
