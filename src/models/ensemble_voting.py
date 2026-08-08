"""
Ensemble voting module for deepfake detection.
Combines predictions from EfficientNetB0, SVM, Random Forest, and KNN classifiers.
"""
import numpy as np
import logging
import joblib
import os
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

# Models that operate on raw image tensors rather than extracted 1280-dim features.
# Anything in this set is routed X_images in predict_proba(); everything else gets X_features.
IMAGE_INPUT_MODELS = {'efficientnetb0'}

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted ensemble classifier combining multiple models.
    Uses soft voting with dynamically adjusted weights based on validation performance.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize the weighted ensemble classifier.

        Args:
            config_path: Path to model configuration YAML file
        """
        self.config_path = config_path
        self.models = {}  # Dictionary to hold individual models
        self.weights = None  # Model weights for voting
        self.is_fitted = False
        self.classes_ = None

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load ensemble configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            ensemble_config = config.get('model', {}).get('ensemble', {})
            self.voting_method = ensemble_config.get('voting', 'soft')
            default_weights = ensemble_config.get('weights', [0.4, 0.2, 0.2, 0.2])
            self.default_weights = np.array(default_weights)
            self.calibration_method = ensemble_config.get('calibration_method', 'sigmoid')

        except Exception as e:
            logger.warning(f"Could not load ensemble config: {e}. Using defaults.")
            self.voting_method = 'soft'
            self.default_weights = np.array([0.4, 0.2, 0.2, 0.2])  # [EfficientNetB0, SVM, RF, KNN]
            self.calibration_method = 'sigmoid'

    def add_model(self, name: str, model):
        """
        Add a model to the ensemble.

        Args:
            name: Name of the model (e.g., 'efficientnetb0', 'svm', 'random_forest', 'knn')
            model: Trained model object with predict_proba method
        """
        self.models[name] = model
        logger.info(f"Added model '{name}' to ensemble")

    def _validate_models(self):
        """Validate that all required models are present."""
        required_models = ['efficientnetb0', 'svm', 'random_forest', 'knn']
        missing_models = [model for model in required_models if model not in self.models]

        if missing_models:
            raise ValueError(f"Missing required models: {missing_models}")

        logger.info(f"All required models present: {list(self.models.keys())}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedEnsembleClassifier':
        """
        Fit the ensemble classifier (this is a placeholder as individual models should be pre-trained).

        Args:
            X: Feature matrix (not used directly, models should be pre-trained)
            y: Target vector (not used directly)

        Returns:
            Self for method chaining
        """
        logger.info("Setting up weighted ensemble classifier")

        # Validate that models are present
        self._validate_models()

        # Set classes from the first model
        first_model = list(self.models.values())[0]
        if hasattr(first_model, 'classes_'):
            self.classes_ = first_model.classes_
        else:
            # For binary classification, assume classes [0, 1]
            self.classes_ = np.array([0, 1])

        # Initialize weights as a name->weight mapping so weights stay attached to
        # the model they belong to even if a model gets dropped later (see predict_proba).
        weight_order = ['efficientnetb0', 'svm', 'random_forest', 'knn']
        self.weights = {
            name: self.default_weights[weight_order.index(name)] if name in weight_order
            else 1.0 / max(len(self.models), 1)
            for name in self.models
        }
        self.is_fitted = True

        logger.info("Weighted ensemble classifier setup completed")
        return self

    def predict_proba(self, X_features: Optional[np.ndarray] = None,
                       X_images=None) -> np.ndarray:
        """
        Predict class probabilities using weighted averaging.

        SVM/Random Forest/KNN take extracted 1280-dim EfficientNetB0 features
        (X_features). The EfficientNetB0 adapter takes raw image tensors
        (X_images). Pass whichever inputs the models you added actually need;
        a model whose required input wasn't supplied is skipped (with a
        warning) rather than silently fed the wrong-shaped array.

        Args:
            X_features: Feature matrix of shape (n_samples, 1280) for the
                sklearn-style models (svm, random_forest, knn).
            X_images: Batch of raw image tensors for the CNN adapter model
                (efficientnetb0), if that model is in the ensemble.

        Returns:
            Weighted average probability estimates, shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Get predictions from each model, routing each to the input it expects
        predictions = []
        used_names = []

        for name, model in self.models.items():
            model_input = X_images if name in IMAGE_INPUT_MODELS else X_features
            if model_input is None:
                logger.warning(
                    f"No input supplied for model '{name}' "
                    f"(expects {'X_images' if name in IMAGE_INPUT_MODELS else 'X_features'}); skipping"
                )
                continue
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(model_input)
                    predictions.append(probs)
                    used_names.append(name)
                else:
                    logger.warning(f"Model {name} does not have predict_proba method")
            except Exception as e:
                logger.error(f"Error getting predictions from model {name}: {e}")

        if not predictions:
            raise ValueError("No valid model predictions obtained")

        # Stack predictions: (n_models, n_samples, n_classes)
        predictions = np.array(predictions)

        # Weights keyed by model name so a dropped/skipped model can never
        # cause a surviving model to be weighted with the wrong entry.
        used_weights = np.array([self.weights[name] for name in used_names])
        weight_sum = np.sum(used_weights)
        if weight_sum <= 0:
            used_weights = np.ones_like(used_weights)
            weight_sum = np.sum(used_weights)

        weighted_predictions = predictions * used_weights.reshape(-1, 1, 1)

        # Sum across models and normalize so probabilities sum to 1
        ensemble_probs = np.sum(weighted_predictions, axis=0) / weight_sum

        return ensemble_probs

    def predict(self, X_features: Optional[np.ndarray] = None, X_images=None) -> np.ndarray:
        """
        Predict class labels using weighted voting.

        Args:
            X_features: Feature matrix of shape (n_samples, 1280) for the
                sklearn-style models.
            X_images: Batch of raw image tensors for the CNN adapter model.

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X_features=X_features, X_images=X_images)
        return np.argmax(probs, axis=1)

    def update_weights(self, val_X_features: Optional[np.ndarray] = None,
                        val_y: np.ndarray = None, val_X_images=None):
        """
        Update model weights based on validation performance.

        Args:
            val_X_features: Validation feature matrix for sklearn-style models
            val_y: Validation target vector
            val_X_images: Validation image batch for the CNN adapter model
        """
        logger.info("Updating model weights based on validation performance")

        individual_accuracies = {}

        for name, model in self.models.items():
            model_input = val_X_images if name in IMAGE_INPUT_MODELS else val_X_features
            try:
                if model_input is not None and hasattr(model, 'predict'):
                    preds = model.predict(model_input)
                    accuracy = np.mean(preds == val_y)
                    individual_accuracies[name] = accuracy
                    logger.info(f"Model {name} validation accuracy: {accuracy:.4f}")
                else:
                    logger.warning(f"Model {name} does not have predict method or input; skipping")
                    individual_accuracies[name] = 0.0
            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
                individual_accuracies[name] = 0.0

        # Convert accuracies to weights (higher accuracy = higher weight)
        total = sum(individual_accuracies.values())
        if total > 0:
            self.weights = {name: acc / total for name, acc in individual_accuracies.items()}
        else:
            # Fall back to default weights if all accuracies are zero
            weight_order = ['efficientnetb0', 'svm', 'random_forest', 'knn']
            self.weights = {
                name: self.default_weights[weight_order.index(name)] if name in weight_order
                else 1.0 / max(len(self.models), 1)
                for name in self.models
            }

        logger.info(f"Updated ensemble weights: {self.weights}")

    def save_ensemble(self, filepath: str):
        """
        Save the ensemble configuration to disk.

        Args:
            filepath: Path to save the ensemble
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ensemble")

        ensemble_data = {
            'models': {name: model for name, model in self.models.items() if hasattr(model, 'save_model')},
            'weights': self.weights,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_,
            'config': {
                'voting_method': self.voting_method,
                'default_weights': self.default_weights.tolist(),
                'calibration_method': self.calibration_method
            }
        }

        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble configuration saved to {filepath}")

        # Save individual models that support it
        for name, model in self.models.items():
            if hasattr(model, 'save_model'):
                model_path = filepath.replace('.pkl', f'_{name}_model.pkl')
                model.save_model(model_path)

    def load_ensemble(self, filepath: str):
        """
        Load the ensemble configuration from disk.

        Args:
            filepath: Path to the saved ensemble
        """
        from src.models.svm_classifier import SVMClassifier
        from src.models.random_forest_classifier import RandomForestClassifier
        from src.models.knn_classifier import KNNClassifier

        ensemble_data = joblib.load(filepath)
        self.weights = ensemble_data['weights']
        self.is_fitted = ensemble_data['is_fitted']
        self.classes_ = ensemble_data['classes_']

        config = ensemble_data.get('config', {})
        self.voting_method = config.get('voting_method', 'soft')
        self.default_weights = np.array(config.get('default_weights', [0.4, 0.2, 0.2, 0.2]))
        self.calibration_method = config.get('calibration_method', 'sigmoid')

        # Load individual models
        model_mapping = {
            'svm': SVMClassifier,
            'random_forest': RandomForestClassifier,
            'knn': KNNClassifier
        }

        for name, model_class in model_mapping.items():
            model_path = filepath.replace('.pkl', f'_{name}_model.pkl')
            if os.path.exists(model_path):
                model = model_class()
                model.load_model(model_path)
                self.models[name] = model
                logger.info(f"Loaded {name} model from {model_path}")

        logger.info(f"Ensemble configuration loaded from {filepath}")


def create_weighted_ensemble(config_path: str = None) -> WeightedEnsembleClassifier:
    """
    Factory function to create a weighted ensemble classifier.

    Args:
        config_path: Path to model configuration YAML file

    Returns:
        WeightedEnsembleClassifier instance
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    return WeightedEnsembleClassifier(config_path)