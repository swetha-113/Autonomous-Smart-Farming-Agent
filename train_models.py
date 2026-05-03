"""
train_models.py
One-shot script to train/retrain all models.
Usage:
    python train_models.py --model all
    python train_models.py --model soil
    python train_models.py --model cnn --data_dir path/to/PlantVillage
"""
import argparse
import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent))
from config import MODELS_DIR


def train_soil():
    logger.info("Training Soil Health Model…")
    from src.models.soil_model import SoilHealthAnalyzer
    analyzer = SoilHealthAnalyzer.__new__(SoilHealthAnalyzer)
    analyzer.model = None
    analyzer.loaded = False
    analyzer.train()
    logger.success("Soil model trained and saved!")


def train_cnn(data_dir: str = None, epochs: int = 20):
    from src.models.cnn_disease_model import CropDiseaseDetector
    detector = CropDiseaseDetector()
    if data_dir:
        logger.info(f"Training CNN on dataset: {data_dir}")
        detector.train(data_dir=data_dir, epochs=epochs)
    else:
        logger.warning(
            "No data_dir provided. Skipping CNN training.\n"
            "Download PlantVillage dataset from:\n"
            "  kaggle datasets download -d abdallahalidev/plantvillage-dataset\n"
            "Then run: python train_models.py --model cnn --data_dir plantvillage/color"
        )


def train_weather():
    logger.info("Fitting Prophet weather forecasting model…")
    from src.models.weather_model import WeatherForecaster
    wf = WeatherForecaster()
    wf.fit_prophet()
    logger.success("Weather forecaster fitted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Farming Agent models")
    parser.add_argument("--model", choices=["all", "soil", "cnn", "weather"],
                        default="all", help="Which model to train")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to PlantVillage dataset for CNN training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for CNN training")
    args = parser.parse_args()

    if args.model in ("all", "soil"):
        train_soil()
    if args.model in ("all", "cnn"):
        train_cnn(args.data_dir, args.epochs)
    if args.model in ("all", "weather"):
        train_weather()

    logger.success("Training complete. Models saved to: " + str(MODELS_DIR))
