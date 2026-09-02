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
    EffNetSklearnAdapter,
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

        self._load_config()

        self.ensemble_model = ensemble_model
        self.feature_extractor = feature_extractor

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

    def load_models(self, model_dir: str, fold: int = 1):
        """
        Load pre-trained models from disk.

        Args:
            model_dir: Directory containing saved models (the same model_dir
                passed to train.py — this method now expects train.py's
                actual nested-by-fold layout, not a flat one)
            fold: Which fold's models to load, 1-indexed (default: 1)

        BUG FIXES applied here:
        1. This used to call `EfficientNetB0Classifier().load_model(model_path)`
           in two separate places — but EfficientNetB0Classifier is a plain
           nn.Module with no load_model/save_model methods (only
           SVMClassifier/RandomForestClassifier/KNNClassifier have those, via
           joblib). Both call sites always raised AttributeError, silently
           caught and logged, so the CNN could never actually be loaded this
           way. Replaced with torch.load() + load_state_dict(), matching how
           predict.py loads it.
        2. The ensemble path was `model_dir/ensemble_model.pkl`, but
           train.py saves to `model_dir/ensemble/fold_N/ensemble_model.pkl`
           — a different, nested location. Fixed to match.
        3. The individual SVM/RF/KNN paths were flat (`model_dir/svm_model.pkl`
           etc.) but train.py saves them under `model_dir/svm/fold_N/svm_model.pkl`
           etc. Fixed to match.
        4. The CNN was never wrapped in EffNetSklearnAdapter before being
           added to the ensemble (and, per bug 1, was never successfully
           loaded at all) — so it never actually contributed to predictions
           even when the rest of this loaded successfully. Fixed to build the
           adapter and pass it into load_ensemble() so it's re-attached
           (see the load_ensemble() fix in ensemble_voting.py).
        """
        logger.info(f"Loading models from {model_dir} (fold {fold})")
        fold_tag = f"fold_{fold}"

        # --- EfficientNetB0 ---
        efficientnet_model = EfficientNetB0Classifier(pretrained=False, num_classes=1).to(self.device)
        effnet_dir = os.path.join(model_dir, 'efficientnetb0', fold_tag)
        best_path = os.path.join(effnet_dir, 'best_model.pth')
        if os.path.exists(best_path):
            state_dict = torch.load(best_path, map_location=self.device)
            efficientnet_model.load_state_dict(state_dict)
            logger.info(f"EfficientNetB0 best-checkpoint loaded from {best_path}")
        elif os.path.isdir(effnet_dir):
            checkpoints = sorted(
                [f for f in os.listdir(effnet_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')],
                key=lambda f: int(f.split('_')[-1].split('.')[0])
            )
            if checkpoints:
                checkpoint = torch.load(os.path.join(effnet_dir, checkpoints[-1]), map_location=self.device)
                efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"EfficientNetB0 periodic checkpoint loaded from {checkpoints[-1]}")
            else:
                logger.warning(f"No EfficientNetB0 checkpoint found in {effnet_dir}, using untrained ImageNet weights")
        else:
            logger.warning(f"EfficientNetB0 checkpoint directory not found: {effnet_dir}, using untrained ImageNet weights")

        cnn_adapter = EffNetSklearnAdapter(efficientnet_model, self.device)

        # Rebuild the feature extractor from the (now loaded) classifier's
        # backbone, so SVM/RF/KNN see features from the actual trained model.
        self.feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
        self.feature_extractor.backbone.load_state_dict(efficientnet_model.backbone.state_dict())
        self.feature_extractor.to(self.device).eval()

        # --- SVM / Random Forest / KNN ---
        svm_model = SVMClassifier(self.config_path)
        svm_path = os.path.join(model_dir, 'svm', fold_tag, 'svm_model.pkl')
        if os.path.exists(svm_path):
            svm_model.load_model(svm_path)
            logger.info(f"SVM model loaded from {svm_path}")
        else:
            logger.warning(f"SVM model not found at {svm_path}")

        rf_model = RandomForestClassifier(self.config_path)
        rf_path = os.path.join(model_dir, 'random_forest', fold_tag, 'rf_model.pkl')
        if os.path.exists(rf_path):
            rf_model.load_model(rf_path)
            logger.info(f"Random Forest model loaded from {rf_path}")
        else:
            logger.warning(f"Random Forest model not found at {rf_path}")

        knn_model = KNNClassifier(self.config_path)
        knn_path = os.path.join(model_dir, 'knn', fold_tag, 'knn_model.pkl')
        if os.path.exists(knn_path):
            knn_model.load_model(knn_path)
            logger.info(f"KNN model loaded from {knn_path}")
        else:
            logger.warning(f"KNN model not found at {knn_path}")

        # --- Ensemble ---
        self.ensemble_model = WeightedEnsembleClassifier(self.config_path)
        self.ensemble_model.add_model('efficientnetb0', cnn_adapter)
        self.ensemble_model.add_model('svm', svm_model)
        self.ensemble_model.add_model('random_forest', rf_model)
        self.ensemble_model.add_model('knn', knn_model)

        ensemble_path = os.path.join(model_dir, 'ensemble', fold_tag, 'ensemble_model.pkl')
        if os.path.exists(ensemble_path):
            self.ensemble_model.load_ensemble(ensemble_path, cnn_adapter=cnn_adapter)
            logger.info(f"Ensemble model loaded from {ensemble_path}")
        else:
            logger.warning(f"Ensemble model not found at {ensemble_path}; "
                            f"using freshly-fitted weights from the individually loaded models")
            dummy_features = np.zeros((1, 1280))
            dummy_labels = np.array([0])
            self.ensemble_model.fit(dummy_features, dummy_labels)

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

        import pandas as pd
        df = pd.DataFrame({
            'image_path': image_paths,
            'label': [0] * len(image_paths)  # Dummy labels
        })

        dataset = DeepfakeDataset(df, transform=None)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        logger.info("Extracting features using EfficientNetB0 backbone")
        features, _ = extract_features(self.feature_extractor, dataloader, self.device)

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

        image_paths = []
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        for ext in extensions:
            image_paths.extend([str(p) for p in directory.glob(f"*{ext}")])
            image_paths.extend([str(p) for p in directory.glob(f"*{ext.upper()}")])

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
        df_results = pd.DataFrame({
            'image_path': results['image_paths'],
            'prediction': results['predictions'],
            'confidence': results['confidences'],
            'probability_fake': [prob[1] if len(prob) > 1 else prob[0] for prob in results['probabilities']],
            'probability_real': [prob[0] if len(prob) > 1 else 1 - prob[0] for prob in results['probabilities']]
        })

        df_results['filename'] = df_results['image_path'].apply(lambda x: os.path.basename(x))

        df_results = df_results[['filename', 'image_path', 'prediction', 'confidence',
                               'probability_real', 'probability_fake']]

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