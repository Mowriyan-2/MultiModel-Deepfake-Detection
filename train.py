#!/usr/bin/env python3
"""
Main training script for Multi-model-deep-fake-detection.
Orchestrates data loading, model training, ensemble creation, and evaluation.
"""
import argparse
import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DeepfakeDatasetProcessor
from src.feature_extraction import (
    EfficientNetB0Classifier,
    EfficientNetB0FeatureExtractor,
    EffNetSklearnAdapter,
)
from src.models.svm_classifier import SVMClassifier
from src.models.random_forest_classifier import RandomForestClassifier
from src.models.knn_classifier import KNNClassifier
from src.models.ensemble_voting import WeightedEnsembleClassifier
from src.training.trainer import ModelTrainer
from src.training.cross_validation import CrossValidator
from src.inference.batch_inference import BatchInferenceEngine
from src.inference.confidence_calibration import ConfidenceCalibrator
from src.utils import (setup_logging, get_device, set_seed, create_directories,
                      load_config, save_config, plot_training_history, format_time,
                      get_system_info, log_system_info)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "model_config.yaml")

def setup_environment(args):
    """Setup logging, device, seeds, and directories."""
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level, log_file=args.log_file)

    # Log system information
    log_system_info()

    # Set random seeds
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Create necessary directories
    directories = [
        args.data_dir,
        args.model_dir,
        args.output_dir,
        os.path.join(args.model_dir, 'efficientnetb0'),
        os.path.join(args.model_dir, 'svm'),
        os.path.join(args.model_dir, 'random_forest'),
        os.path.join(args.model_dir, 'knn'),
        os.path.join(args.model_dir, 'ensemble'),
        os.path.join(args.output_dir, 'logs'),
        os.path.join(args.output_dir, 'plots'),
        os.path.join(args.output_dir, 'results')
    ]
    create_directories(directories)

    return device

def load_configurations(args):
    """Load model and training configurations."""
    # Load model config
    model_config_path = args.model_config or DEFAULT_CONFIG_PATH
    model_config = load_config(model_config_path)

    # Load training config if provided
    training_config = {}
    if args.training_config:
        training_config = load_config(args.training_config)
        # Merge with model config
        model_config.update(training_config)

    return model_config

def prepare_data(args, device):
    """Prepare datasets for training."""
    logger = logging.getLogger(__name__)
    logger.info("Preparing datasets...")

    # Initialize data processor
    data_processor = DeepfakeDatasetProcessor(args.model_config)

    # Load dataset based on argument
    if args.dataset == 'faceforensics':
        logger.info(f"Loading FaceForensics++ dataset with compression {args.compression}")
        df = data_processor.load_faceforensics_data(args.data_path, args.compression)
    elif args.dataset == 'celebdf':
        logger.info("Loading Celeb-DF dataset")
        df = data_processor.load_celeb_df_data(args.data_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    logger.info(f"Dataset loaded: {len(df)} samples "
               f"({len(df[df.label==0])} real, {len(df[df.label==1])} fake)")

    # Create stratified splits
    logger.info(f"Creating {args.n_folds}-fold stratified splits")
    splits = data_processor.create_stratified_splits(df, args.n_folds, args.seed)

    # Save splits if requested
    if args.save_splits:
        splits_dir = os.path.join(args.output_dir, 'splits')
        data_processor.save_splits(splits, df, splits_dir)
        logger.info(f"Splits saved to {splits_dir}")

    return df, splits, data_processor

def train_individual_models(args, device, df, splits, data_processor, model_config):
    """Train individual models (EfficientNetB0, SVM, RF, KNN)."""
    logger = logging.getLogger(__name__)
    logger.info("Training individual models...")

    # Import dataset class
    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader

    # Initialize models
    efficientnet_model = EfficientNetB0Classifier(
        pretrained=model_config.get('efficientnetb0', {}).get('pretrained', True),
        num_classes=model_config.get('efficientnetb0', {}).get('num_classes', 1),
        dropout_rate=model_config.get('efficientnetb0', {}).get('dropout_rate', 0.3)
    ).to(device)

    svm_model = SVMClassifier(args.model_config)
    rf_model = RandomForestClassifier(args.model_config)
    knn_model = KNNClassifier(args.model_config)

    # Store training histories
    histories = {}

    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(splits):
        if args.fold is not None and fold != args.fold:
            continue  # Skip if specific fold requested

        logger.info(f"=== Training Fold {fold+1}/{args.n_folds} ===")

        # Split data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Create datasets
        train_dataset = DeepfakeDataset(train_df, transform=None)
        val_dataset = DeepfakeDataset(val_df, transform=None)

        # Create data loaders
        batch_size = model_config.get('training', {}).get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Initialize trainers
        effnet_trainer = ModelTrainer(efficientnet_model, device=device, config_path=args.model_config)
        # For traditional ML models, we'll extract features first

        # Train EfficientNetB0
        logger.info("Training EfficientNetB0...")
        start_time = time.time()
        effnet_history = effnet_trainer.fit(train_loader, val_loader,
                                          epochs=model_config.get('training', {}).get('epochs', 50),
                                          save_dir=os.path.join(args.model_dir, f'efficientnetb0/fold_{fold+1}') if args.save_models else None)
        effnet_time = time.time() - start_time
        logger.info(f"EfficientNetB0 training completed in {format_time(effnet_time)}")

        # Extract features for traditional ML models
        logger.info("Extracting features for traditional ML models...")
        from src.feature_extraction import extract_features

        # SVM/RF/KNN need the pooled 1280-dim EfficientNetB0 features, not the
        # classifier's logits — build a feature extractor that shares the
        # just-trained backbone weights rather than feeding it the classifier.
        feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
        feature_extractor.backbone.load_state_dict(efficientnet_model.backbone.state_dict())
        feature_extractor.to(device).eval()

        # Get features for training
        train_features, train_labels = extract_features(feature_extractor, train_loader, device)
        val_features, val_labels = extract_features(feature_extractor, val_loader, device)

        # Train SVM
        logger.info("Training SVM...")
        start_time = time.time()
        svm_model.fit(train_features, train_labels)
        svm_time = time.time() - start_time
        logger.info(f"SVM training completed in {format_time(svm_time)}")

        # Train Random Forest
        logger.info("Training Random Forest...")
        start_time = time.time()
        rf_model.fit(train_features, train_labels)
        rf_time = time.time() - start_time
        logger.info(f"Random Forest training completed in {format_time(rf_time)}")

        # Train KNN
        logger.info("Training KNN...")
        start_time = time.time()
        knn_model.fit(train_features, train_labels)
        knn_time = time.time() - start_time
        logger.info(f"KNN training completed in {format_time(knn_time)}")

        # Store histories
        histories[f'fold_{fold+1}'] = {
            'efficientnetb0': effnet_history,
            'svm_time': svm_time,
            'rf_time': rf_time,
            'knn_time': knn_time,
            'effnet_time': effnet_time
        }

        # Save models if requested
        if args.save_models:
            # Save EfficientNetB0
            effnet_path = os.path.join(args.model_dir, f'efficientnetb0/fold_{fold+1}', 'model_best.pth')
            if os.path.exists(effnet_path):
                # Already saved by trainer
                pass

            # Save traditional ML models
            svm_model.save_model(os.path.join(args.model_dir, f'svm/fold_{fold+1}', 'svm_model.pkl'))
            rf_model.save_model(os.path.join(args.model_dir, f'random_forest/fold_{fold+1}', 'rf_model.pkl'))
            knn_model.save_model(os.path.join(args.model_dir, f'knn/fold_{fold+1}', 'knn_model.pkl'))

        logger.info(f"Fold {fold+1} completed")

    return efficientnet_model, feature_extractor, svm_model, rf_model, knn_model, histories

def create_ensemble(args, device, efficientnet_model, feature_extractor, svm_model, rf_model, knn_model, df, splits, data_processor):
    """Create and train weighted ensemble."""
    logger = logging.getLogger(__name__)
    logger.info("Creating weighted ensemble...")

    # Initialize ensemble
    ensemble = WeightedEnsembleClassifier(args.model_config)

    # The raw EfficientNetB0Classifier has no predict_proba/sklearn-style
    # interface, and it expects raw images rather than the 1280-dim features
    # SVM/RF/KNN use — wrap it in the adapter before adding it to the ensemble.
    cnn_adapter = EffNetSklearnAdapter(efficientnet_model, device)

    # Add models
    ensemble.add_model('efficientnetb0', cnn_adapter)
    ensemble.add_model('svm', svm_model)
    ensemble.add_model('random_forest', rf_model)
    ensemble.add_model('knn', knn_model)

    # Create a combined dataset for ensemble weight optimization
    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features

    # Use first fold for weight optimization (or combine all folds)
    train_idx, val_idx = splits[0]  # Use first fold
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DeepfakeDataset(train_df, transform=None)
    val_dataset = DeepfakeDataset(val_df, transform=None)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Extract features (for SVM/RF/KNN) using the feature extractor, not the classifier
    logger.info("Extracting features for ensemble training...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)

    # Fit ensemble (this sets up the structure)
    ensemble.fit(train_features, train_labels)

    # Update weights based on validation performance
    logger.info("Optimizing ensemble weights...")
    ensemble.update_weights(val_X_features=val_features, val_y=val_labels)

    logger.info(f"Final ensemble weights: {ensemble.weights}")

    # Save ensemble
    if args.save_models:
        ensemble_dir = os.path.join(args.model_dir, 'ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble.save_ensemble(os.path.join(ensemble_dir, 'ensemble_model.pkl'))
        logger.info(f"Ensemble saved to {ensemble_dir}")

    return ensemble

def evaluate_model(args, device, ensemble, feature_extractor, df, splits, data_processor):
    """Evaluate the ensemble model."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating ensemble model...")

    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features
    from src.utils import calculate_metrics
    import numpy as np

    # Use all folds for evaluation, or specific fold if requested
    folds_to_evaluate = [args.fold] if args.fold is not None else range(len(splits))

    fold_results = []

    for fold in folds_to_evaluate:
        logger.info(f"Evaluating Fold {fold+1}/{len(splits)}")
        train_idx, val_idx = splits[fold]
        # For evaluation, we use the validation set
        val_df = df.iloc[val_idx].reset_index(drop=True)

        val_dataset = DeepfakeDataset(val_df, transform=None)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Extract features using the same feature extractor (trained backbone)
        # that was used for training — not a freshly re-initialized model,
        # which would silently evaluate against untrained ImageNet features.
        # Note: evaluation here only exercises the SVM/RF/KNN path (X_features);
        # the CNN adapter is skipped with a logged warning since no raw image
        # batch is supplied — extend this call with X_images= to include it.
        val_features, val_labels = extract_features(feature_extractor, val_loader, device)

        # Make predictions
        logger.info("Making predictions...")
        val_predictions = ensemble.predict(X_features=val_features)
        val_probabilities = ensemble.predict_proba(X_features=val_features)

        # Calculate metrics
        metrics = calculate_metrics(val_labels, val_predictions, val_probabilities)
        metrics['fold'] = fold + 1

        fold_results.append(metrics)

        logger.info(f"Fold {fold+1} Results:")
        for metric_name, value in metrics.items():
            if metric_name != 'fold':
                logger.info(f"  {metric_name}: {value:.4f}")

    # Calculate average results
    if len(fold_results) > 1:
        avg_metrics = {}
        for key in fold_results[0].keys():
            if key != 'fold':
                values = [result[key] for result in fold_results]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)

        logger.info("Average Results Across Folds:")
        for metric_name, value in avg_metrics.items():
            if not metric_name.endswith('_std'):
                std_value = avg_metrics.get(f'{metric_name}_std', 0)
                logger.info(f"  {metric_name}: {value:.4f} ± {std_value:.4f}")

        # Save results
        results_path = os.path.join(args.output_dir, 'results', 'evaluation_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump({
                'fold_results': fold_results,
                'average_results': avg_metrics
            }, f, indent=2)
        logger.info(f"Evaluation results saved to {results_path}")

    return fold_results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Multi-model Deepfake Detection System')

    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['faceforensics', 'celebdf'],
                       help='Dataset to train on')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--compression', type=str, default='c23',
                       choices=['c23', 'c40'],
                       help='Compression level for FaceForensics++ (default: c23)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for data storage (default: ./data)')

    # Model arguments
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory to save models (default: ./models)')
    parser.add_argument('--model_config', type=str,
                       default=DEFAULT_CONFIG_PATH,
                       help='Path to model configuration YAML file')
    parser.add_argument('--training_config', type=str,
                       help='Path to training configuration YAML file (optional)')

    # Training arguments
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--fold', type=int,
                       help='Specific fold to train (if not specified, trains all folds)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models to disk')
    parser.add_argument('--save_splits', action='store_true',
                       help='Save train/validation splits to disk')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory for output files (default: ./output)')
    parser.add_argument('--log_file', type=str,
                       help='Path to log file (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    # Evaluation arguments
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate existing models, do not train')
    parser.add_argument('--resume_from', type=str,
                       help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    try:
        # Setup environment
        device = setup_environment(args)

        # Load configurations
        model_config = load_configurations(args)

        # Override config with command line arguments
        if 'training' not in model_config:
            model_config['training'] = {}
        model_config['training']['epochs'] = args.epochs
        model_config['training']['batch_size'] = args.batch_size
        model_config['training']['learning_rate'] = args.learning_rate

        logger = logging.getLogger(__name__)
        logger.info("="*60)
        logger.info("Multi-model Deepfake Detection Training Started")
        logger.info("="*60)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Data path: {args.data_path}")
        if args.dataset == 'faceforensics':
            logger.info(f"Compression: {args.compression}")
        logger.info(f"Number of folds: {args.n_folds}")
        if args.fold is not None:
            logger.info(f"Training fold: {args.fold + 1}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Device: {device}")
        logger.info("="*60)

        start_time = time.time()

        if args.evaluate_only:
            logger.info("Evaluation mode: Loading existing models...")
            # TODO: Implement model loading and evaluation
            logger.warning("Evaluation-only mode not fully implemented yet")
        else:
            # Prepare data
            df, splits, data_processor = prepare_data(args, device)

            # Train individual models
            efficientnet_model, feature_extractor, svm_model, rf_model, knn_model, histories = train_individual_models(
                args, device, df, splits, data_processor, model_config)

            # Create ensemble
            ensemble = create_ensemble(
                args, device, efficientnet_model, feature_extractor, svm_model, rf_model, knn_model,
                df, splits, data_processor)

            # Evaluate ensemble
            evaluate_model(args, device, ensemble, feature_extractor, df, splits, data_processor)

        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"Training completed successfully in {format_time(total_time)}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()