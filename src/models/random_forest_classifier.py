"""
Random Forest Classifier for deepfake detection.
Uses features extracted from EfficientNetB0 backbone.
"""
import numpy as np
import logging
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

class RandomForestClassifier:
    """
    Random Forest classifier with probability calibration for deepfake detection.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize Random Forest classifier.

        Args:
            config_path: Path to model configuration YAML file
        """
        self.config_path = config_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load Random Forest configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            rf_config = config.get('model', {}).get('random_forest', {})
            self.n_estimators = rf_config.get('n_estimators', 100)
            self.max_depth = rf_config.get('max_depth', 20)
            self.min_samples_split = rf_config.get('min_samples_split', 2)
            self.min_samples_leaf = rf_config.get('min_samples_leaf', 1)

        except Exception as e:
            logger.warning(f"Could not load Random Forest config: {e}. Using defaults.")
            self.n_estimators = 100
            self.max_depth = 20
            self.min_samples_split = 2
            self.min_samples_leaf = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Fit the Random Forest classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting Random Forest classifier with {X.shape[0]} samples and {X.shape[1]} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create Random Forest classifier
        base_rf = SKRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        # Calibrate probabilities
        self.model = CalibratedClassifierCV(base_rf, method='sigmoid', cv=3)
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info("Random Forest classifier fitting completed")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Probability estimates for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        joblib.dump(model_data, filepath)
        logger.info(f"Random Forest model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']

        config = model_data.get('config', {})
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 20)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.min_samples_leaf = config.get('min_samples_leaf', 1)

        logger.info(f"Random Forest model loaded from {filepath}")


def create_random_forest_classifier(config_path: str = None) -> RandomForestClassifier:
    """
    Factory function to create a Random Forest classifier.

    Args:
        config_path: Path to model configuration YAML file

    Returns:
        RandomForestClassifier instance
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    return RandomForestClassifier(config_path)