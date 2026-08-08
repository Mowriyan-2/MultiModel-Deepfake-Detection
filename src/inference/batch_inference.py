"""
Batch inference engine for deepfake detection.
Handles processing large batches of images with confidence calibration.
"""
import numpy as np
import torch
import logging
import os
import time
from typing import Tuple, List, Dict, Any, Optional, Union
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
from src.data_preprocessing import DeepfakeDataset
from src.feature_extraction import (
    EfficientNetB0Classifier,
    EfficientNetB0FeatureExtractor,
    extract_features,
)
from src.models.svm_classifier import SVMClassifier
from src.models.random_forest_classifier import RandomForestClassifier
from src.models.knn_classifier import KNNClassifier
from src.models.ensemble_voting import WeightedEnsembleClassifier
from src.training.trainer import ModelTrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

class BatchInferenceEngine:
    """
    Batch inference engine for deepfake detection with confidence calibration.
    """

    def __init__(self, ensemble_model: WeightedEnsembleClassifier = None,
                 feature_extractor: EfficientNetB0FeatureExtractor = None,
                 device: torch.device = None,
                 config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize batch inference engine.

        Args:
            ensemble_model: Trained WeightedEnsembleClassifier
            feature_extractor: EfficientNetB0 feature extractor (must be an
                EfficientNetB0FeatureExtractor — the SVM/RF/KNN models were
                trained on its 1280-dim pooled output, not classifier logits)
            device: Computing device (cuda/cpu)
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self._load_config()

        # Initialize components
        self.ensemble_model = ensemble_model
        self.feature_extractor = feature_extractor

        # If models not provided, create placeholder (will need to be loaded separately)
        if self.ensemble_model is None:
            self.ensemble_model = WeightedEnsembleClassifier(config_path)
        if self.feature_extractor is None:
            self.feature_extractor = EfficientNetB0FeatureExtractor(pretrained=True)

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        logger.info(f"BatchInferenceEngine initialized on {self.device}")

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.batch_size = config.get('training', {}).get('batch_size', 32)
            self.image_size = tuple(config.get('preprocessing', {}).get('image_size', [224, 224]))
            self.normalize_mean = config.get('preprocessing', {}).get('normalize_mean', [0.485, 0.456, 0.406])
            self.normalize_std = config.get('preprocessing', {}).get('normalize_std', [0.229, 0.224, 0.225])

        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}. Using defaults.")
            self.batch_size = 32
            self.image_size = (224, 224)
            self.normalize_mean = [0.485, 0.456, 0.406]
            self.normalize_std = [0.229, 0.224, 0.225]

    def load_models(self, model_dir: str):
        """
        Load pre-trained models from disk.

        Args:
            model_dir: Directory containing saved models
        """
        logger.info(f"Loading models from {model_dir}")

        # Load ensemble configuration
        ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
        if os.path.exists(ensemble_path):
            self.ensemble_model.load_ensemble(ensemble_path)
            logger.info("Ensemble model loaded")
        else:
            logger.warning(f"Ensemble model not found at {ensemble_path}")

        # Load individual models
        model_mapping = {
            'svm': ('svm_model.pkl', SVMClassifier),
            'random_forest': ('rf_model.pkl', RandomForestClassifier),
            'knn': ('knn_model.pkl', KNNClassifier),
            'efficientnetb0': ('efficientnetb0_model.pkl', EfficientNetB0Classifier)
        }

        for name, (filename, model_class) in model_mapping.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = model_class()
                    model.load_model(model_path)
                    self.ensemble_model.add_model(name, model)
                    logger.info(f"{name} model loaded")
                except Exception as e:
                    logger.error(f"Failed to load {name} model: {e}")
            else:
                logger.warning(f"{name} model not found at {model_path}")

        # Load feature extractor if not already loaded
        if self.feature_extractor is None or not hasattr(self.feature_extractor, 'backbone'):
            effnet_path = os.path.join(model_dir, 'efficientnetb0_model.pkl')
            if os.path.exists(effnet_path):
                classifier = EfficientNetB0Classifier()
                classifier.load_model(effnet_path)
                self.feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
                self.feature_extractor.backbone.load_state_dict(classifier.backbone.state_dict())
                self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
                logger.info("Feature extractor loaded (weights copied from trained classifier backbone)")
            else:
                logger.warning("Feature extractor model not found, using default ImageNet weights")

    def predict_batch(self, image_paths: List[str], return_features: bool = False) -> Dict[str, Any]:
        """
        Perform batch inference on a list of image paths.

        Args:
            image_paths: List of paths to input images
            return_features: Whether to return extracted features

        Returns:
            Dictionary containing predictions, confidences, and optionally features
        """
        logger.info(f"Starting batch inference on {len(image_paths)} images")

        start_time = time.time()

        # Create dataset and dataloader
        # Create a simple dataframe for the images
        import pandas as pd
        df = pd.DataFrame({
            'image_path': image_paths,
            'label': [0] * len(image_paths)  # Dummy labels
        })

        dataset = DeepfakeDataset(df, transform=None)  # Will use default preprocessing
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # Extract features using EfficientNetB0 backbone
        logger.info("Extracting features using EfficientNetB0 backbone")
        features, _ = extract_features(self.feature_extractor, dataloader, self.device)

        # Make predictions using ensemble
        logger.info("Making predictions using weighted ensemble")
        if self.ensemble_model.is_fitted:
            probabilities = self.ensemble_model.predict_proba(X_features=features)
            predictions = self.ensemble_model.predict(X_features=features)
            confidences = np.max(probabilities, axis=1)  # Maximum probability as confidence
        else:
            logger.warning("Ensemble model not fitted, using dummy predictions")
            predictions = np.zeros(len(features))
            confidences = np.ones(len(features)) * 0.5
            probabilities = np.column_stack([1 - confidences, confidences])

        inference_time = time.time() - start_time
        logger.info(f"Batch inference completed in {inference_time:.2f} seconds")

        # Prepare results
        results = {
            'image_paths': image_paths,
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'probabilities': probabilities.tolist(),
            'inference_time': inference_time,
            'avg_time_per_image': inference_time / len(image_paths) if len(image_paths) > 0 else 0
        }

        if return_features:
            results['features'] = features.tolist()

        return results

    def predict_directory(self, directory_path: str, extensions: List[str] = None,
                         return_features: bool = False) -> Dict[str, Any]:
        """
        Perform batch inference on all images in a directory.

        Args:
            directory_path: Path to directory containing images
            extensions: List of file extensions to process (default: common image formats)
            return_features: Whether to return extracted features

        Returns:
            Dictionary containing predictions and metadata
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Find all image files
        image_paths = []
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        for ext in extensions:
            image_paths.extend([str(p) for p in directory.glob(f"*{ext}")])
            image_paths.extend([str(p) for p in directory.glob(f"*{ext.upper()}")])

        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))

        logger.info(f"Found {len(image_paths)} images in {directory_path}")

        return self.predict_batch(image_paths, return_features)

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save inference results to CSV file.

        Args:
            results: Results dictionary from predict_batch or predict_directory
            output_path: Path to save CSV file
        """
        # Create DataFrame for easy CSV export
        df_results = pd.DataFrame({
            'image_path': results['image_paths'],
            'prediction': results['predictions'],
            'confidence': results['confidences'],
            'probability_fake': [prob[1] if len(prob) > 1 else prob[0] for prob in results['probabilities']],
            'probability_real': [prob[0] if len(prob) > 1 else 1 - prob[0] for prob in results['probabilities']]
        })

        # Add filename column
        df_results['filename'] = df_results['image_path'].apply(lambda x: os.path.basename(x))

        # Reorder columns
        df_results = df_results[['filename', 'image_path', 'prediction', 'confidence',
                               'probability_real', 'probability_fake']]

        # Save to CSV
        df_results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def visualize_predictions(self, results: Dict[str, Any],
                            save_dir: str = None,
                            max_visualizations: int = 10) -> List[str]:
        """
        Create visualizations for predictions (requires GradCAM integration).

        Args:
            results: Results dictionary from inference
            save_dir: Directory to save visualizations
            max_visualizations: Maximum number of images to visualize

        Returns:
            List of paths to saved visualization images
        """
        # This would require integrating with GradCAM - placeholder for now
        logger.info("Visualization functionality would require GradCAM integration")
        return []

def create_batch_inference_engine(config_path: str = None) -> BatchInferenceEngine:
    """
    Factory function to create a batch inference engine.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        BatchInferenceEngine instance
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    return BatchInferenceEngine(config_path=config_path)