"""
src/models/cnn_disease_model.py
CNN-based crop disease detection using transfer learning (MobileNetV2).
"""
import os
import numpy as np
from pathlib import Path
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. Using mock CNN model.")

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import CNN_MODEL_PATH, DISEASE_CLASSES, MODELS_DIR


class CropDiseaseDetector:
    """
    CNN-based crop disease classifier.
    Uses MobileNetV2 backbone with fine-tuned head for 38 disease classes.
    Falls back to a mock predictor if TF is unavailable.
    """

    IMG_SIZE = (224, 224)
    NUM_CLASSES = len(DISEASE_CLASSES)

    def __init__(self):
        self.model = None
        self.loaded = False
        self._try_load()

    # ── Build ──────────────────────────────────────────────────────────────
    def build_model(self) -> "tf.keras.Model":
        """Construct transfer-learning model (MobileNetV2 + custom head)."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required to build the model.")

        base = MobileNetV2(
            input_shape=(*self.IMG_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )
        base.trainable = False  # freeze backbone initially

        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)

        model = Model(inputs=base.input, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        logger.info(f"CNN model built — {model.count_params():,} parameters")
        return model

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, data_dir: str, epochs: int = 20, batch_size: int = 32):
        """
        Train on PlantVillage dataset.
        data_dir should contain subdirectories per class.
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow unavailable — cannot train.")
            return

        train_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            validation_split=0.2
        )

        train_data = train_gen.flow_from_directory(
            data_dir, target_size=self.IMG_SIZE,
            batch_size=batch_size, class_mode="categorical",
            subset="training"
        )
        val_data = train_gen.flow_from_directory(
            data_dir, target_size=self.IMG_SIZE,
            batch_size=batch_size, class_mode="categorical",
            subset="validation"
        )

        self.model = self.build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint(
                str(CNN_MODEL_PATH), save_best_only=True
            )
        ]

        history = self.model.fit(
            train_data, validation_data=val_data,
            epochs=epochs, callbacks=callbacks
        )

        # Fine-tune top layers
        logger.info("Fine-tuning top 30 layers of MobileNetV2…")
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:-30]:
            layer.trainable = False
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.fit(
            train_data, validation_data=val_data,
            epochs=10, callbacks=callbacks
        )
        logger.success(f"Model saved → {CNN_MODEL_PATH}")
        return history

    # ── Inference ──────────────────────────────────────────────────────────
    def predict(self, image_array: np.ndarray) -> dict:
        """
        Predict disease from a (H, W, 3) uint8 or float image.
        Returns dict with top-3 predictions.
        """
        if not self.loaded:
            return self._mock_predict(image_array)

        img = self._preprocess(image_array)
        probs = self.model.predict(img, verbose=0)[0]
        top3_idx = np.argsort(probs)[::-1][:3]

        return {
            "top_prediction": DISEASE_CLASSES[top3_idx[0]],
            "confidence": float(probs[top3_idx[0]]),
            "top3": [
                {"class": DISEASE_CLASSES[i], "confidence": float(probs[i])}
                for i in top3_idx
            ],
            "is_healthy": "healthy" in DISEASE_CLASSES[top3_idx[0]].lower()
        }

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize + normalise to [0,1]."""
        from PIL import Image
        if TF_AVAILABLE:
            img = tf.image.resize(img, self.IMG_SIZE).numpy()
        else:
            img = np.array(Image.fromarray(img).resize(self.IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, 0)

    def _try_load(self):
        """Load saved model if it exists."""
        if not TF_AVAILABLE:
            return
        if CNN_MODEL_PATH.exists():
            try:
                self.model = tf.keras.models.load_model(str(CNN_MODEL_PATH))
                self.loaded = True
                logger.info(f"CNN model loaded from {CNN_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load CNN model: {e}")
        else:
            logger.info("No pre-trained CNN found. Using mock predictor.")

    def _mock_predict(self, image_array: np.ndarray) -> dict:
        """Deterministic mock for demo/testing."""
        import hashlib
        h = int(hashlib.md5(image_array.tobytes()[:1000]).hexdigest(), 16)
        idx = h % len(DISEASE_CLASSES)
        top_class = DISEASE_CLASSES[idx]
        conf = 0.72 + (h % 25) / 100

        return {
            "top_prediction": top_class,
            "confidence": min(conf, 0.97),
            "top3": [
                {"class": DISEASE_CLASSES[(idx + i) % len(DISEASE_CLASSES)],
                 "confidence": max(conf - i * 0.15, 0.05)}
                for i in range(3)
            ],
            "is_healthy": "healthy" in top_class.lower(),
            "mock": True
        }


if __name__ == "__main__":
    # Quick smoke test
    detector = CropDiseaseDetector()
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = detector.predict(dummy)
    print("Prediction:", result)
