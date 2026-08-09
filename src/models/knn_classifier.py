"""
KNN Classifier for deepfake detection.
Uses features extracted from EfficientNetB0 backbone.
"""
import numpy as np
import logging
import joblib
import os
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import yaml

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

logger = logging.getLogger(__name__)

class KNNClassifier:
    """
    KNN classifier with probability calibration for deepfake detection.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize KNN classifier.

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
        """Load KNN configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            knn_config = config.get('model', {}).get('knn', {})
            self.n_neighbors = knn_config.get('n_neighbors', 5)
            self.weights = knn_config.get('weights', 'distance')
            self.algorithm = knn_config.get('algorithm', 'auto')

        except Exception as e:
            logger.warning(f"Could not load KNN config: {e}. Using defaults.")
            self.n_neighbors = 5
            self.weights = 'distance'
            self.algorithm = 'auto'

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting KNN classifier with {X.shape[0]} samples and {X.shape[1]} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create KNN classifier
        base_knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm
        )

        # Calibrate probabilities
        self.model = CalibratedClassifierCV(base_knn, method='sigmoid', cv=3)
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info("KNN classifier fitting completed")
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
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        joblib.dump(model_data, filepath)
        logger.info(f"KNN model saved to {filepath}")

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
        self.n_neighbors = config.get('n_neighbors', 5)
        self.weights = config.get('weights', 'distance')
        self.algorithm = config.get('algorithm', 'auto')

        logger.info(f"KNN model loaded from {filepath}")


def create_knn_classifier(config_path: str = None) -> KNNClassifier:
    """
    Factory function to create a KNN classifier.

    Args:
        config_path: Path to model configuration YAML file

    Returns:
        KNNClassifier instance
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    return KNNClassifier(config_path)