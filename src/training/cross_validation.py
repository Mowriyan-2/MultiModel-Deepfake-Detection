"""
5-fold stratified cross-validation for deepfake detection models.
"""
import numpy as np
import torch
import logging
import os
import time
from typing import Tuple, List, Dict, Any, Optional
import yaml
from sklearn.model_selection import StratifiedKFold
from src.data_preprocessing import DeepfakeDatasetProcessor, DeepfakeDataset
from src.feature_extraction import EfficientNetB0Classifier
from src.training.trainer import ModelTrainer
from torch.utils.data import DataLoader
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

class CrossValidator:
    """
    5-fold stratified cross-validation for model evaluation.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42,
                 config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize cross validator.

        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            config_path: Path to configuration file
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.config_path = config_path

        # Load configuration
        self._load_config()

        # Initialize data processor
        self.data_processor = DeepfakeDatasetProcessor(config_path)

        logger.info(f"CrossValidator initialized for {n_folds}-fold CV")

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.training_config = config.get('training', {})
            self.model_config = config.get('model', {})

        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}. Using defaults.")
            self.training_config = {
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'batch_size': 32,
                'epochs': 50
            }
            self.model_config = {
                'efficientnetb0': {
                    'pretrained': True,
                    'num_classes': 1,
                    'dropout_rate': 0.3
                }
            }

    def validate_dataset(self, dataset_name: str, data_path: str,
                        compression: str = None, save_dir: str = None) -> Dict[str, Any]:
        """
        Perform 5-fold stratified cross-validation on a dataset.

        Args:
            dataset_name: Name of the dataset ('faceforensics', 'celebdf', etc.)
            data_path: Path to the dataset
            compression: Compression level (for FaceForensics++)
            save_dir: Directory to save fold models and results

        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Starting {self.n_folds}-fold cross-validation on {dataset_name}")

        # Load dataset
        if dataset_name.lower() == 'faceforensics':
            df = self.data_processor.load_faceforensics_data(data_path, compression)
        elif dataset_name.lower() == 'celebdf':
            df = self.data_processor.load_celeb_df_data(data_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Create stratified folds
        splits = self.data_processor.create_stratified_splits(df, self.n_folds, self.random_state)

        # Save splits if directory provided
        if save_dir is not None:
            splits_dir = os.path.join(save_dir, 'splits')
            self.data_processor.save_splits(splits, df, splits_dir)

        # Initialize results storage
        fold_results = []
        fold_models = []

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Starting Fold {fold+1}/{self.n_folds}")

            # Split data
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            # Create datasets
            train_dataset = DeepfakeDataset(train_df, transform=None)  # Will use default transforms
            val_dataset = DeepfakeDataset(val_df, transform=None)

            # Create data loaders
            batch_size = self.training_config.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Initialize model
            model = EfficientNetB0Classifier(
                pretrained=self.model_config.get('efficientnetb0', {}).get('pretrained', True),
                num_classes=self.model_config.get('efficientnetb0', {}).get('num_classes', 1),
                dropout_rate=self.model_config.get('efficientnetb0', {}).get('dropout_rate', 0.3)
            )

            # Initialize trainer
            trainer = ModelTrainer(model, config_path=self.config_path)

            # Train model
            fold_save_dir = None
            if save_dir is not None:
                fold_save_dir = os.path.join(save_dir, f'fold_{fold+1}')
                os.makedirs(fold_save_dir, exist_ok=True)

            start_time = time.time()
            history = trainer.fit(train_loader, val_loader,
                                epochs=self.training_config.get('epochs', 50),
                                save_dir=fold_save_dir)
            fold_time = time.time() - start_time

            # Evaluate on validation set
            val_loss, val_acc = trainer.validate_epoch(val_loader)

            # Store results
            fold_result = {
                'fold': fold + 1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_time': fold_time,
                'epochs_trained': len(history['train_losses']),
                'final_train_loss': history['train_losses'][-1] if history['train_losses'] else 0,
                'final_train_acc': history['train_accs'][-1] if history['train_accs'] else 0,
                'history': history
            }

            fold_results.append(fold_result)
            fold_models.append(model)

            logger.info(f"Fold {fold+1} completed - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Calculate aggregate results
        val_losses = [result['val_loss'] for result in fold_results]
        val_accs = [result['val_acc'] for result in fold_results]

        cv_results = {
            'dataset': dataset_name,
            'n_folds': self.n_folds,
            'fold_results': fold_results,
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_acc': np.mean(val_accs),
            'std_val_acc': np.std(val_accs),
            'fold_models': fold_models
        }

        logger.info(f"Cross-validation completed:")
        logger.info(f"  Mean Val Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
        logger.info(f"  Mean Val Acc: {cv_results['mean_val_acc']:.4f} ± {cv_results['std_val_acc']:.4f}")

        return cv_results

def run_cross_validation(dataset_name: str, data_path: str,
                        compression: str = None, save_dir: str = None,
                        n_folds: int = 5, random_state: int = 42,
                        config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Convenience function to run 5-fold stratified cross-validation.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset
        compression: Compression level (for FaceForensics++)
        save_dir: Directory to save results
        n_folds: Number of folds
        random_state: Random seed
        config_path: Path to configuration file

    Returns:
        Cross-validation results dictionary
    """
    validator = CrossValidator(n_folds=n_folds, random_state=random_state, config_path=config_path)
    return validator.validate_dataset(dataset_name, data_path, compression, save_dir)