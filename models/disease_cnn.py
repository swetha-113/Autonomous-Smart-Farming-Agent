"""
models/disease_cnn.py - CNN model for crop disease detection using ResNet50 transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from loguru import logger

from config.settings import IMAGE_SIZE, NUM_DISEASE_CLASSES, DISEASE_CLASSES, CNN_MODEL_PATH


class CropDiseaseModel(nn.Module):
    """ResNet50-based CNN for crop disease classification."""

    def __init__(self, num_classes: int = NUM_DISEASE_CLASSES, pretrained: bool = True):
        super(CropDiseaseModel, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze early layers (transfer learning)
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < 100:
                param.requires_grad = False

        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class DiseaseDetector:
    """High-level interface for disease detection from images."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CropDiseaseModel()
        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No saved model found. Using untrained model (run training first).")
            self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        """
        Predict disease from a PIL Image.
        Returns: dict with disease_class, confidence, top_predictions
        """
        try:
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Transform
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            pred_idx = predicted.item()
            conf_val = confidence.item()

            # Top-5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
            top5 = [
                {
                    "class": DISEASE_CLASSES[idx.item()],
                    "probability": prob.item(),
                    "formatted": DISEASE_CLASSES[idx.item()].replace("___", " - ").replace("_", " ")
                }
                for idx, prob in zip(top5_indices[0], top5_probs[0])
            ]

            disease_name = DISEASE_CLASSES[pred_idx]
            is_healthy = "healthy" in disease_name.lower()
            plant, condition = disease_name.split("___") if "___" in disease_name else (disease_name, "Unknown")

            return {
                "success": True,
                "plant": plant.replace("_", " "),
                "condition": condition.replace("_", " "),
                "disease_class": disease_name,
                "confidence": conf_val,
                "confidence_pct": f"{conf_val * 100:.1f}%",
                "is_healthy": is_healthy,
                "severity": self._estimate_severity(conf_val, is_healthy),
                "top_predictions": top5,
                "requires_treatment": not is_healthy and conf_val > 0.5
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"success": False, "error": str(e)}

    def _estimate_severity(self, confidence: float, is_healthy: bool) -> str:
        if is_healthy:
            return "None"
        if confidence > 0.85:
            return "High"
        elif confidence > 0.65:
            return "Medium"
        else:
            return "Low"

    def predict_from_path(self, image_path: str) -> dict:
        """Load image from path and predict."""
        image = Image.open(image_path)
        return self.predict(image)


class ModelTrainer:
    """Trainer for the CNN model with PlantVillage-style dataset."""

    def __init__(self, data_dir: str, save_path: str = CNN_MODEL_PATH, epochs: int = 20, batch_size: int = 32):
        self.data_dir = data_dir
        self.save_path = save_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform

    def train(self):
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, random_split

        train_tf, val_tf = self.get_transforms()

        full_dataset = ImageFolder(self.data_dir, transform=train_tf)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        val_ds.dataset.transform = val_tf

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=4)

        model = CropDiseaseModel(num_classes=len(full_dataset.classes))
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_acc = 0
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            scheduler.step(1 - val_acc)
            history["train_loss"].append(running_loss / len(train_loader))
            history["val_acc"].append(val_acc)
            logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classes': full_dataset.classes,
                    'val_acc': val_acc,
                    'epoch': epoch
                }, self.save_path)
                logger.info(f"  ✓ Saved best model (acc={val_acc:.4f})")

        return history
