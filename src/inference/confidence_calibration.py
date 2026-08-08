"""
Confidence calibration for deepfake detection predictions.
Implements Platt scaling and isotonic regression for better probability estimates.
"""
import numpy as np
import logging
import joblib
import os
from typing import Tuple, List, Optional, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """
    Confidence calibration for classifier predictions.
    Supports Platt scaling (logistic regression) and isotonic regression.
    """

    def __init__(self, method: str = 'platt'):
        """
        Initialize confidence calibrator.

        Args:
            method: Calibration method ('platt' or 'isotonic')
        """
        self.method = method.lower()
        self.calibrator = None
        self.is_fitted = False

        if self.method not in ['platt', 'isotonic']:
            raise ValueError("Method must be 'platt' or 'isotonic'")

        logger.info(f"ConfidenceCalibrator initialized with {self.method} method")

    def fit(self, predictions: np.ndarray, true_labels: np.ndarray) -> 'ConfidenceCalibrator':
        """
        Fit the calibrator on validation predictions.

        Args:
            predictions: Model predictions (probabilities or scores) of shape (n_samples,)
                       For binary classification, should be probability of positive class
            true_labels: True binary labels of shape (n_samples,)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting confidence calibrator with {len(predictions)} samples")

        # Ensure predictions are in correct format
        if predictions.ndim > 1:
            if predictions.shape[1] == 2:
                # Binary classification with two columns
                predictions = predictions[:, 1]  # Probability of positive class
            else:
                # Multi-class - use max probability
                predictions = np.max(predictions, axis=1)

        # Clip predictions to avoid extreme values
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)

        if self.method == 'platt':
            # Platt scaling: logistic regression on predictions
            self.calibrator = LogisticRegression()
            self.calibrator.fit(predictions.reshape(-1, 1), true_labels)
        elif self.method == 'isotonic':
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(predictions, true_labels)

        self.is_fitted = True
        logger.info(f"Confidence calibrator fitted using {self.method} method")
        return self

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions to improve confidence estimates.

        Args:
            predictions: Model predictions (probabilities or scores) of shape (n_samples,)

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibrating predictions")

        logger.info(f"Calibrating {len(predictions)} predictions")

        # Ensure predictions are in correct format
        if predictions.ndim > 1:
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
            else:
                predictions = np.max(predictions, axis=1)

        # Clip predictions to avoid extreme values
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)

        if self.method == 'platt':
            # Platt scaling: use logistic regression to get calibrated probabilities
            calibrated_probs = self.calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            # Isotonic regression
            calibrated_probs = self.calibrator.transform(predictions)

        return calibrated_probs

    def calibrate_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate probability estimates for multi-class classification.

        Args:
            probabilities: Probability estimates of shape (n_samples, n_classes)

        Returns:
            Calibrated probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibrating probabilities")

        logger.info(f"Calibrating probability estimates for {probabilities.shape[0]} samples")

        if probabilities.shape[1] == 2:
            # Binary classification - calibrate positive class probability
            pos_probs = probabilities[:, 1]
            calibrated_pos_probs = self.calibrate(pos_probs)
            calibrated_probs = np.column_stack([1 - calibrated_pos_probs, calibrated_pos_probs])
        else:
            # Multi-class - calibrate each class separately (simplified approach)
            calibrated_probs = np.zeros_like(probabilities)
            for i in range(probabilities.shape[1]):
                class_probs = probabilities[:, i]
                calibrated_class_probs = self.calibrate(class_probs)
                calibrated_probs[:, i] = calibrated_class_probs

            # Renormalize to ensure probabilities sum to 1
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            calibrated_probs = calibrated_probs / row_sums

        return calibrated_probs

    def save_calibrator(self, filepath: str):
        """
        Save the calibrator to disk.

        Args:
            filepath: Path to save the calibrator
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")

        calibrator_data = {
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted
        }

        joblib.dump(calibrator_data, filepath)
        logger.info(f"Confidence calibrator saved to {filepath}")

    def load_calibrator(self, filepath: str):
        """
        Load a calibrator from disk.

        Args:
            filepath: Path to the saved calibrator
        """
        calibrator_data = joblib.load(filepath)
        self.method = calibrator_data['method']
        self.calibrator = calibrator_data['calibrator']
        self.is_fitted = calibrator_data['is_fitted']

        logger.info(f"Confidence calibrator loaded from {filepath}")


def create_confidence_calibrator(method: str = 'platt') -> ConfidenceCalibrator:
    """
    Factory function to create a confidence calibrator.

    Args:
        method: Calibration method ('platt' or 'isotonic')

    Returns:
        ConfidenceCalibrator instance
    """
    return ConfidenceCalibrator(method)


def calibrate_ensemble_predictions(ensemble_predictions: np.ndarray,
                                 true_labels: np.ndarray = None,
                                 method: str = 'platt') -> Tuple[np.ndarray, ConfidenceCalibrator]:
    """
    Convenience function to calibrate ensemble predictions.

    Args:
        ensemble_predictions: Ensemble predictions of shape (n_samples, n_classes)
        true_labels: True labels for fitting (if None, returns uncalibrated predictions)
        method: Calibration method ('platt' or 'isotonic')

    Returns:
        Tuple of (calibrated_predictions, calibrator)
    """
    calibrator = create_confidence_calibrator(method)

    if true_labels is not None:
        calibrator.fit(ensemble_predictions, true_labels)
        calibrated_probs = calibrator.calibrate_proba(ensemble_predictions)
        return calibrated_probs, calibrator
    else:
        # Return uncalibrated predictions and unfitted calibrator
        return ensemble_predictions, calibrator