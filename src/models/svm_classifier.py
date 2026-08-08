"""
SVM Classifier for deepfake detection.
Uses features extracted from EfficientNetB0 backbone.
"""
import numpy as np
import logging
import joblib
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import yaml

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

logger = logging.getLogger(__name__)

class SVMClassifier:
    """
    SVM classifier with probability calibration for deepfake detection.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize SVM classifier.

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
        """Load SVM configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            svm_config = config.get('model', {}).get('svm', {})
            self.kernel = svm_config.get('kernel', 'rbf')
            self.C = svm_config.get('C', 1.0)
            self.gamma = svm_config.get('gamma', 'scale')
            self.probability = svm_config.get('probability', True)

        except Exception as e:
            logger.warning(f"Could not load SVM config: {e}. Using defaults.")
            self.kernel = 'rbf'
            self.C = 1.0
            self.gamma = 'scale'
            self.probability = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Fit the SVM classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting SVM classifier with {X.shape[0]} samples and {X.shape[1]} features")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create SVM classifier
        base_svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=42
        )

        # Calibrate probabilities
        self.model = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info("SVM classifier fitting completed")
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
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'probability': self.probability
            }
        }

        joblib.dump(model_data, filepath)
        logger.info(f"SVM model saved to {filepath}")

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
        self.kernel = config.get('kernel', 'rbf')
        self.C = config.get('C', 1.0)
        self.gamma = config.get('gamma', 'scale')
        self.probability = config.get('probability', True)

        logger.info(f"SVM model loaded from {filepath}")


def create_svm_classifier(config_path: str = None) -> SVMClassifier:
    """
    Factory function to create an SVM classifier.

    Args:
        config_path: Path to model configuration YAML file

    Returns:
        SVMClassifier instance
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    return SVMClassifier(config_path)