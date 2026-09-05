#!/usr/bin/env python3
"""
Main training script for Multi-model-deep-fake-detection.
Orchestrates data loading, model training, ensemble creation, and evaluation.
"""
import argparse
import os
import sys
import logging
import json
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
from src.inference.batch_inference import BatchInferenceEngine
from src.inference.confidence_calibration import ConfidenceCalibrator
from src.utils import (setup_logging, get_device, set_seed, create_directories,
                      load_config, save_config, plot_training_history, format_time,
                      get_system_info, log_system_info)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "model_config.yaml")


def setup_environment(args):
    """Setup logging, device, seeds, and directories."""
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level, log_file=args.log_file)
    log_system_info()
    set_seed(args.seed)
    device = get_device()

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
    model_config_path = args.model_config or DEFAULT_CONFIG_PATH
    model_config = load_config(model_config_path)

    training_config = {}
    if args.training_config:
        training_config = load_config(args.training_config)
    model_config.update(training_config)

    return model_config


def resolved_model_config_path(args):
    """Resolve the config PATH (not the loaded dict) for constructors that need it."""
    return args.model_config or DEFAULT_CONFIG_PATH


def prepare_data(args, device):
    """Prepare datasets for training."""
    logger = logging.getLogger(__name__)
    logger.info("Preparing datasets...")

    data_processor = DeepfakeDatasetProcessor(resolved_model_config_path(args))

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

    # BUG FIX: splits are now video-grouped (see DeepfakeDatasetProcessor.
    # create_stratified_splits) instead of frame-level, so frames from one
    # video can no longer land in both the train and validation side of a fold.
    logger.info(f"Creating {args.n_folds}-fold video-grouped stratified splits")
    splits = data_processor.create_stratified_splits(df, args.n_folds, args.seed)

    if args.save_splits:
        splits_dir = os.path.join(args.output_dir, 'splits')
        data_processor.save_splits(splits, df, splits_dir)
        logger.info(f"Splits saved to {splits_dir}")

    return df, splits, data_processor


def train_one_fold(args, device, df, splits, fold, model_config):
    """
    Train EfficientNetB0 + SVM + RF + KNN for a single fold, using freshly
    constructed models.

    BUG FIX: previously all four models were constructed once outside the
    fold loop and reused/retrained across every fold without ever being
    reset. Since standard k-fold means fold i's held-out validation samples
    are part of every OTHER fold's training set, reusing the same model
    object meant that by fold 5 the model had already been trained (via
    backprop) on fold 5's own "held out" samples during folds 1-4 — silently
    inflating every fold's reported validation accuracy after the first.
    Constructing fresh models here, once per fold, closes that leak.
    """
    logger = logging.getLogger(__name__)
    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features

    train_idx, val_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DeepfakeDataset(train_df, transform=None)
    val_dataset = DeepfakeDataset(val_df, transform=None)

    batch_size = model_config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Fresh models every fold.
    effnet_cfg = model_config.get('model', {}).get('efficientnetb0', {})
    efficientnet_model = EfficientNetB0Classifier(
        pretrained=effnet_cfg.get('pretrained', True),
        num_classes=effnet_cfg.get('num_classes', 1),
        dropout_rate=effnet_cfg.get('dropout_rate', 0.3)
    ).to(device)
    svm_model = SVMClassifier(resolved_model_config_path(args))
    rf_model = RandomForestClassifier(resolved_model_config_path(args))
    knn_model = KNNClassifier(resolved_model_config_path(args))

    effnet_trainer = ModelTrainer(efficientnet_model, device=device, config_path=resolved_model_config_path(args))

    logger.info("Training EfficientNetB0...")
    start_time = time.time()
    save_dir = os.path.join(args.model_dir, f'efficientnetb0/fold_{fold+1}') if args.save_models else None
    effnet_history = effnet_trainer.fit(
        train_loader, val_loader,
        epochs=model_config.get('training', {}).get('epochs', 50),
        save_dir=save_dir
    )
    effnet_time = time.time() - start_time
    logger.info(f"EfficientNetB0 training completed in {format_time(effnet_time)}")

    # SVM/RF/KNN need the pooled 1280-dim EfficientNetB0 features, not the
    # classifier's logits — build a feature extractor sharing the
    # just-trained (and now best-checkpoint-restored) backbone weights.
    feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
    feature_extractor.backbone.load_state_dict(efficientnet_model.backbone.state_dict())
    feature_extractor.to(device).eval()

    logger.info("Extracting features for traditional ML models...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)

    logger.info("Training SVM...")
    start_time = time.time()
    svm_model.fit(train_features, train_labels)
    logger.info(f"SVM training completed in {format_time(time.time() - start_time)}")

    logger.info("Training Random Forest...")
    start_time = time.time()
    rf_model.fit(train_features, train_labels)
    logger.info(f"Random Forest training completed in {format_time(time.time() - start_time)}")

    logger.info("Training KNN...")
    start_time = time.time()
    knn_model.fit(train_features, train_labels)
    logger.info(f"KNN training completed in {format_time(time.time() - start_time)}")

    if args.save_models:
        svm_model.save_model(os.path.join(args.model_dir, f'svm/fold_{fold+1}', 'svm_model.pkl'))
        rf_model.save_model(os.path.join(args.model_dir, f'random_forest/fold_{fold+1}', 'rf_model.pkl'))
        knn_model.save_model(os.path.join(args.model_dir, f'knn/fold_{fold+1}', 'knn_model.pkl'))
    # BUG FIX: the old code repeated this exact save block a second time,
    # unconditionally, right after the fold's try/except — ignoring the
    # `if args.save_models:` guard entirely and silently saving to disk even
    # when --save_models wasn't passed. It also duplicated the "Fold N
    # completed" log line, which is why earlier runs showed it twice per
    # fold. That duplicate block has been removed; this is the only save.

    logger.info(f"Fold {fold+1} completed")

    return {
        'efficientnet_model': efficientnet_model,
        'feature_extractor': feature_extractor,
        'svm_model': svm_model,
        'rf_model': rf_model,
        'knn_model': knn_model,
        'effnet_history': effnet_history,
        # ModelTrainer.fit() already computes this — reuse it for ensemble
        # weighting instead of re-running inference over raw images just to
        # score the CNN again (see create_ensemble_for_fold below).
        'efficientnet_val_accuracy': effnet_history.get('best_val_acc'),
    }


def create_ensemble_for_fold(args, device, fold_models, df, splits, fold):
    """
    Build and weight the ensemble for one fold.

    BUG FIX (wrong split): previously this always used splits[0] regardless
    of which fold's models were passed in, so every fold except fold 0 was
    weighted against the wrong validation set. Now takes the fold index
    explicitly and uses splits[fold].

    BUG FIX (CNN gets zero weight): previously update_weights() was only
    ever given val_X_features, so the CNN (which needs raw images, not
    features) always hit the "no input; skipping" path and got weight 0 —
    meaning the most expensive model to train contributed nothing to the
    final ensemble. Now its own best validation accuracy (already computed
    during training) is passed in directly via precomputed_accuracies,
    without needing to re-run inference over a large batch of raw images.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating weighted ensemble for fold {fold+1}...")

    efficientnet_model = fold_models['efficientnet_model']
    feature_extractor = fold_models['feature_extractor']
    svm_model = fold_models['svm_model']
    rf_model = fold_models['rf_model']
    knn_model = fold_models['knn_model']
    efficientnet_val_accuracy = fold_models['efficientnet_val_accuracy']

    ensemble = WeightedEnsembleClassifier(resolved_model_config_path(args))

    cnn_adapter = EffNetSklearnAdapter(efficientnet_model, device)
    ensemble.add_model('efficientnetb0', cnn_adapter)
    ensemble.add_model('svm', svm_model)
    ensemble.add_model('random_forest', rf_model)
    ensemble.add_model('knn', knn_model)

    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features

    train_idx, val_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DeepfakeDataset(train_df, transform=None)
    val_dataset = DeepfakeDataset(val_df, transform=None)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info("Extracting features for ensemble training...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)

    ensemble.fit(train_features, train_labels)

    logger.info("Optimizing ensemble weights...")
    ensemble.update_weights(
        val_X_features=val_features, val_y=val_labels,
        precomputed_accuracies=(
            {'efficientnetb0': efficientnet_val_accuracy}
            if efficientnet_val_accuracy is not None else None
        )
    )

    logger.info(f"Final ensemble weights for fold {fold+1}: {ensemble.weights}")

    if args.save_models:
        ensemble_dir = os.path.join(args.model_dir, 'ensemble', f'fold_{fold+1}')
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble.save_ensemble(os.path.join(ensemble_dir, 'ensemble_model.pkl'))
        logger.info(f"Ensemble saved to {ensemble_dir}")

    return ensemble, val_features, val_labels


def evaluate_fold(ensemble, val_features, val_labels, fold, output_dir=None):
    """Evaluate one fold's ensemble on its own (correctly-matched) validation split.

    If output_dir is given, also saves the raw labels/predictions/probabilities
    to a per-fold .npz file — evaluation_results.json only stores the summary
    metrics (accuracy, precision, etc.), not the raw values needed to draw a
    confusion matrix or ROC curve, so this is what you'd load afterward for
    those visualizations.
    """
    logger = logging.getLogger(__name__)
    from src.utils import calculate_metrics

    logger.info(f"Evaluating Fold {fold+1}...")
    val_predictions = ensemble.predict(X_features=val_features)
    val_probabilities = ensemble.predict_proba(X_features=val_features)

    metrics = calculate_metrics(val_labels, val_predictions, val_probabilities)
    metrics['fold'] = fold + 1

    logger.info(f"Fold {fold+1} Results:")
    for metric_name, value in metrics.items():
        if metric_name != 'fold':
            logger.info(f"  {metric_name}: {value:.4f}")

    if output_dir is not None:
        predictions_dir = os.path.join(output_dir, 'results', 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        npz_path = os.path.join(predictions_dir, f'fold_{fold+1}_predictions.npz')
        np.savez(
            npz_path,
            val_labels=val_labels,
            val_predictions=val_predictions,
            val_probabilities=val_probabilities,
        )
        logger.info(f"Raw predictions saved to {npz_path} (for confusion matrix / ROC curve plotting)")

    return metrics


def parse_fold_spec(fold_spec, n_folds):
    """
    Parse --fold into a list of 0-indexed fold numbers.
    Accepts: None (-> all folds), "2", "3,4", or "3-4".
    """
    if fold_spec is None:
        return list(range(n_folds))
    fold_spec = fold_spec.strip()
    folds = set()
    for part in fold_spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            folds.update(range(int(start), int(end) + 1))
        else:
            folds.add(int(part))
    return sorted(folds)


def merge_fold_results(output_dir, new_fold_results):
    """
    Merge newly-computed fold results into any existing evaluation_results.json
    in output_dir, so resuming a subset of folds after a crash doesn't erase
    the scores already recorded for folds that finished earlier. A fold number
    present in both the existing file and new_fold_results is replaced by the
    new value (covers re-running a fold you want to redo).
    """
    results_path = os.path.join(output_dir, 'results', 'evaluation_results.json')
    existing_fold_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing_fold_results = json.load(f).get('fold_results', [])

    merged = {r['fold']: r for r in existing_fold_results}
    for r in new_fold_results:
        merged[r['fold']] = r
    all_fold_results = [merged[k] for k in sorted(merged.keys())]

    avg_metrics = {}
    if all_fold_results:
        for key in all_fold_results[0].keys():
            if key != 'fold':
                values = [result[key] for result in all_fold_results]
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values))

    with open(results_path, 'w') as f:
        json.dump({
            'fold_results': all_fold_results,
            'average_results': avg_metrics
        }, f, indent=2)

    return all_fold_results, avg_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Multi-model Deepfake Detection System')

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

    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory to save models (default: ./models)')
    parser.add_argument('--model_config', type=str,
                       default=DEFAULT_CONFIG_PATH,
                       help='Path to model configuration YAML file')
    parser.add_argument('--training_config', type=str,
                       help='Path to training configuration YAML file (optional)')

    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--fold', type=str,
                       help='Specific fold(s) to train, 0-indexed. Accepts a single '
                            'number ("2"), a comma-separated list ("3,4"), or a range '
                            '("3-4"). If not specified, trains all folds. Results for '
                            'the folds you run here are MERGED into any existing '
                            'evaluation_results.json in --output_dir, so you can '
                            'resume after a crash without losing earlier folds\' scores.')
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

    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory for output files (default: ./output)')
    parser.add_argument('--log_file', type=str,
                       help='Path to log file (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    try:
        device = setup_environment(args)
        model_config = load_configurations(args)

        if 'training' not in model_config:
            model_config['training'] = {}
        model_config['training']['epochs'] = args.epochs
        model_config['training']['batch_size'] = args.batch_size
        model_config['training']['learning_rate'] = args.learning_rate

        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Multi-model Deepfake Detection Training Started")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Data path: {args.data_path}")
        if args.dataset == 'faceforensics':
            logger.info(f"Compression: {args.compression}")
        logger.info(f"Number of folds: {args.n_folds}")
        folds_to_run = parse_fold_spec(args.fold, args.n_folds)
        logger.info(f"Folds to run this invocation: {[f+1 for f in folds_to_run]}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Device: {device}")
        logger.info("=" * 60)

        start_time = time.time()

        df, splits, data_processor = prepare_data(args, device)

        fold_results = []

        for fold in folds_to_run:
            logger.info(f"=== Fold {fold+1}/{args.n_folds} ===")
            try:
                fold_models = train_one_fold(args, device, df, splits, fold, model_config)
                ensemble, val_features, val_labels = create_ensemble_for_fold(
                    args, device, fold_models, df, splits, fold
                )
                metrics = evaluate_fold(ensemble, val_features, val_labels, fold, output_dir=args.output_dir)
                fold_results.append(metrics)
            except Exception as e:
                logger.error(f"Fold {fold+1} failed: {e}", exc_info=True)
                continue

        # Merge this run's fold results into any existing evaluation_results.json
        # (see merge_fold_results docstring) rather than overwriting it — this is
        # what makes it safe to resume a crashed multi-fold run with just the
        # remaining folds, e.g. --fold 3-4, without losing folds 0-2's scores.
        if fold_results:
            all_fold_results, avg_metrics = merge_fold_results(args.output_dir, fold_results)

            logger.info("Average Results Across All Folds Recorded So Far:")
            for metric_name, value in avg_metrics.items():
                if not metric_name.endswith('_std'):
                    std_value = avg_metrics.get(f'{metric_name}_std', 0)
                    logger.info(f"  {metric_name}: {value:.4f} \u00b1 {std_value:.4f}")

            results_path = os.path.join(args.output_dir, 'results', 'evaluation_results.json')
            logger.info(f"Evaluation results saved to {results_path} "
                        f"({len(all_fold_results)} fold(s) recorded total)")
        else:
            logger.warning("No folds completed successfully — no results to report.")

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training completed successfully in {format_time(total_time)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Main training script for Multi-model-deep-fake-detection.
Orchestrates data loading, model training, ensemble creation, and evaluation.
"""
import argparse
import os
import sys
import logging
import json
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
from src.inference.batch_inference import BatchInferenceEngine
from src.inference.confidence_calibration import ConfidenceCalibrator
from src.utils import (setup_logging, get_device, set_seed, create_directories,
                      load_config, save_config, plot_training_history, format_time,
                      get_system_info, log_system_info)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "model_config.yaml")


def setup_environment(args):
    """Setup logging, device, seeds, and directories."""
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level, log_file=args.log_file)
    log_system_info()
    set_seed(args.seed)
    device = get_device()

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
    model_config_path = args.model_config or DEFAULT_CONFIG_PATH
    model_config = load_config(model_config_path)

    training_config = {}
    if args.training_config:
        training_config = load_config(args.training_config)
    model_config.update(training_config)

    return model_config


def resolved_model_config_path(args):
    """Resolve the config PATH (not the loaded dict) for constructors that need it."""
    return args.model_config or DEFAULT_CONFIG_PATH


def prepare_data(args, device):
    """Prepare datasets for training."""
    logger = logging.getLogger(__name__)
    logger.info("Preparing datasets...")

    data_processor = DeepfakeDatasetProcessor(resolved_model_config_path(args))

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

    # BUG FIX: splits are now video-grouped (see DeepfakeDatasetProcessor.
    # create_stratified_splits) instead of frame-level, so frames from one
    # video can no longer land in both the train and validation side of a fold.
    logger.info(f"Creating {args.n_folds}-fold video-grouped stratified splits")
    splits = data_processor.create_stratified_splits(df, args.n_folds, args.seed)

    if args.save_splits:
        splits_dir = os.path.join(args.output_dir, 'splits')
        data_processor.save_splits(splits, df, splits_dir)
        logger.info(f"Splits saved to {splits_dir}")

    return df, splits, data_processor


def train_one_fold(args, device, df, splits, fold, model_config):
    """
    Train EfficientNetB0 + SVM + RF + KNN for a single fold, using freshly
    constructed models.

    BUG FIX: previously all four models were constructed once outside the
    fold loop and reused/retrained across every fold without ever being
    reset. Since standard k-fold means fold i's held-out validation samples
    are part of every OTHER fold's training set, reusing the same model
    object meant that by fold 5 the model had already been trained (via
    backprop) on fold 5's own "held out" samples during folds 1-4 — silently
    inflating every fold's reported validation accuracy after the first.
    Constructing fresh models here, once per fold, closes that leak.
    """
    logger = logging.getLogger(__name__)
    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features

    train_idx, val_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DeepfakeDataset(train_df, transform=None)
    val_dataset = DeepfakeDataset(val_df, transform=None)

    batch_size = model_config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Fresh models every fold.
    effnet_cfg = model_config.get('model', {}).get('efficientnetb0', {})
    efficientnet_model = EfficientNetB0Classifier(
        pretrained=effnet_cfg.get('pretrained', True),
        num_classes=effnet_cfg.get('num_classes', 1),
        dropout_rate=effnet_cfg.get('dropout_rate', 0.3)
    ).to(device)
    svm_model = SVMClassifier(resolved_model_config_path(args))
    rf_model = RandomForestClassifier(resolved_model_config_path(args))
    knn_model = KNNClassifier(resolved_model_config_path(args))

    effnet_trainer = ModelTrainer(efficientnet_model, device=device, config_path=resolved_model_config_path(args))

    logger.info("Training EfficientNetB0...")
    start_time = time.time()
    save_dir = os.path.join(args.model_dir, f'efficientnetb0/fold_{fold+1}') if args.save_models else None
    effnet_history = effnet_trainer.fit(
        train_loader, val_loader,
        epochs=model_config.get('training', {}).get('epochs', 50),
        save_dir=save_dir
    )
    effnet_time = time.time() - start_time
    logger.info(f"EfficientNetB0 training completed in {format_time(effnet_time)}")

    # SVM/RF/KNN need the pooled 1280-dim EfficientNetB0 features, not the
    # classifier's logits — build a feature extractor sharing the
    # just-trained (and now best-checkpoint-restored) backbone weights.
    feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
    feature_extractor.backbone.load_state_dict(efficientnet_model.backbone.state_dict())
    feature_extractor.to(device).eval()

    logger.info("Extracting features for traditional ML models...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)

    logger.info("Training SVM...")
    start_time = time.time()
    svm_model.fit(train_features, train_labels)
    logger.info(f"SVM training completed in {format_time(time.time() - start_time)}")

    logger.info("Training Random Forest...")
    start_time = time.time()
    rf_model.fit(train_features, train_labels)
    logger.info(f"Random Forest training completed in {format_time(time.time() - start_time)}")

    logger.info("Training KNN...")
    start_time = time.time()
    knn_model.fit(train_features, train_labels)
    logger.info(f"KNN training completed in {format_time(time.time() - start_time)}")

    if args.save_models:
        svm_model.save_model(os.path.join(args.model_dir, f'svm/fold_{fold+1}', 'svm_model.pkl'))
        rf_model.save_model(os.path.join(args.model_dir, f'random_forest/fold_{fold+1}', 'rf_model.pkl'))
        knn_model.save_model(os.path.join(args.model_dir, f'knn/fold_{fold+1}', 'knn_model.pkl'))
    # BUG FIX: the old code repeated this exact save block a second time,
    # unconditionally, right after the fold's try/except — ignoring the
    # `if args.save_models:` guard entirely and silently saving to disk even
    # when --save_models wasn't passed. It also duplicated the "Fold N
    # completed" log line, which is why earlier runs showed it twice per
    # fold. That duplicate block has been removed; this is the only save.

    logger.info(f"Fold {fold+1} completed")

    return {
        'efficientnet_model': efficientnet_model,
        'feature_extractor': feature_extractor,
        'svm_model': svm_model,
        'rf_model': rf_model,
        'knn_model': knn_model,
        'effnet_history': effnet_history,
        # ModelTrainer.fit() already computes this — reuse it for ensemble
        # weighting instead of re-running inference over raw images just to
        # score the CNN again (see create_ensemble_for_fold below).
        'efficientnet_val_accuracy': effnet_history.get('best_val_acc'),
    }


def create_ensemble_for_fold(args, device, fold_models, df, splits, fold):
    """
    Build and weight the ensemble for one fold.

    BUG FIX (wrong split): previously this always used splits[0] regardless
    of which fold's models were passed in, so every fold except fold 0 was
    weighted against the wrong validation set. Now takes the fold index
    explicitly and uses splits[fold].

    BUG FIX (CNN gets zero weight): previously update_weights() was only
    ever given val_X_features, so the CNN (which needs raw images, not
    features) always hit the "no input; skipping" path and got weight 0 —
    meaning the most expensive model to train contributed nothing to the
    final ensemble. Now its own best validation accuracy (already computed
    during training) is passed in directly via precomputed_accuracies,
    without needing to re-run inference over a large batch of raw images.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating weighted ensemble for fold {fold+1}...")

    efficientnet_model = fold_models['efficientnet_model']
    feature_extractor = fold_models['feature_extractor']
    svm_model = fold_models['svm_model']
    rf_model = fold_models['rf_model']
    knn_model = fold_models['knn_model']
    efficientnet_val_accuracy = fold_models['efficientnet_val_accuracy']

    ensemble = WeightedEnsembleClassifier(resolved_model_config_path(args))

    cnn_adapter = EffNetSklearnAdapter(efficientnet_model, device)
    ensemble.add_model('efficientnetb0', cnn_adapter)
    ensemble.add_model('svm', svm_model)
    ensemble.add_model('random_forest', rf_model)
    ensemble.add_model('knn', knn_model)

    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features

    train_idx, val_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = DeepfakeDataset(train_df, transform=None)
    val_dataset = DeepfakeDataset(val_df, transform=None)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info("Extracting features for ensemble training...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)

    ensemble.fit(train_features, train_labels)

    logger.info("Optimizing ensemble weights...")
    ensemble.update_weights(
        val_X_features=val_features, val_y=val_labels,
        precomputed_accuracies=(
            {'efficientnetb0': efficientnet_val_accuracy}
            if efficientnet_val_accuracy is not None else None
        )
    )

    logger.info(f"Final ensemble weights for fold {fold+1}: {ensemble.weights}")

    if args.save_models:
        ensemble_dir = os.path.join(args.model_dir, 'ensemble', f'fold_{fold+1}')
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble.save_ensemble(os.path.join(ensemble_dir, 'ensemble_model.pkl'))
        logger.info(f"Ensemble saved to {ensemble_dir}")

    return ensemble, val_features, val_labels


def evaluate_fold(ensemble, val_features, val_labels, fold, output_dir=None):
    """Evaluate one fold's ensemble on its own (correctly-matched) validation split.

    If output_dir is given, also saves the raw labels/predictions/probabilities
    to a per-fold .npz file — evaluation_results.json only stores the summary
    metrics (accuracy, precision, etc.), not the raw values needed to draw a
    confusion matrix or ROC curve, so this is what you'd load afterward for
    those visualizations.
    """
    logger = logging.getLogger(__name__)
    from src.utils import calculate_metrics

    logger.info(f"Evaluating Fold {fold+1}...")
    val_predictions = ensemble.predict(X_features=val_features)
    val_probabilities = ensemble.predict_proba(X_features=val_features)

    metrics = calculate_metrics(val_labels, val_predictions, val_probabilities)
    metrics['fold'] = fold + 1

    logger.info(f"Fold {fold+1} Results:")
    for metric_name, value in metrics.items():
        if metric_name != 'fold':
            logger.info(f"  {metric_name}: {value:.4f}")

    if output_dir is not None:
        predictions_dir = os.path.join(output_dir, 'results', 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        npz_path = os.path.join(predictions_dir, f'fold_{fold+1}_predictions.npz')
        np.savez(
            npz_path,
            val_labels=val_labels,
            val_predictions=val_predictions,
            val_probabilities=val_probabilities,
        )
        logger.info(f"Raw predictions saved to {npz_path} (for confusion matrix / ROC curve plotting)")

    return metrics


def parse_fold_spec(fold_spec, n_folds):
    """
    Parse --fold into a list of 0-indexed fold numbers.
    Accepts: None (-> all folds), "2", "3,4", or "3-4".
    """
    if fold_spec is None:
        return list(range(n_folds))
    fold_spec = fold_spec.strip()
    folds = set()
    for part in fold_spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            folds.update(range(int(start), int(end) + 1))
        else:
            folds.add(int(part))
    return sorted(folds)


def merge_fold_results(output_dir, new_fold_results):
    """
    Merge newly-computed fold results into any existing evaluation_results.json
    in output_dir, so resuming a subset of folds after a crash doesn't erase
    the scores already recorded for folds that finished earlier. A fold number
    present in both the existing file and new_fold_results is replaced by the
    new value (covers re-running a fold you want to redo).
    """
    results_path = os.path.join(output_dir, 'results', 'evaluation_results.json')
    existing_fold_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing_fold_results = json.load(f).get('fold_results', [])

    merged = {r['fold']: r for r in existing_fold_results}
    for r in new_fold_results:
        merged[r['fold']] = r
    all_fold_results = [merged[k] for k in sorted(merged.keys())]

    avg_metrics = {}
    if all_fold_results:
        for key in all_fold_results[0].keys():
            if key != 'fold':
                values = [result[key] for result in all_fold_results]
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values))

    with open(results_path, 'w') as f:
        json.dump({
            'fold_results': all_fold_results,
            'average_results': avg_metrics
        }, f, indent=2)

    return all_fold_results, avg_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Multi-model Deepfake Detection System')

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

    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory to save models (default: ./models)')
    parser.add_argument('--model_config', type=str,
                       default=DEFAULT_CONFIG_PATH,
                       help='Path to model configuration YAML file')
    parser.add_argument('--training_config', type=str,
                       help='Path to training configuration YAML file (optional)')

    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--fold', type=str,
                       help='Specific fold(s) to train, 0-indexed. Accepts a single '
                            'number ("2"), a comma-separated list ("3,4"), or a range '
                            '("3-4"). If not specified, trains all folds. Results for '
                            'the folds you run here are MERGED into any existing '
                            'evaluation_results.json in --output_dir, so you can '
                            'resume after a crash without losing earlier folds\' scores.')
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

    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory for output files (default: ./output)')
    parser.add_argument('--log_file', type=str,
                       help='Path to log file (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    try:
        device = setup_environment(args)
        model_config = load_configurations(args)

        if 'training' not in model_config:
            model_config['training'] = {}
        model_config['training']['epochs'] = args.epochs
        model_config['training']['batch_size'] = args.batch_size
        model_config['training']['learning_rate'] = args.learning_rate

        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Multi-model Deepfake Detection Training Started")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Data path: {args.data_path}")
        if args.dataset == 'faceforensics':
            logger.info(f"Compression: {args.compression}")
        logger.info(f"Number of folds: {args.n_folds}")
        folds_to_run = parse_fold_spec(args.fold, args.n_folds)
        logger.info(f"Folds to run this invocation: {[f+1 for f in folds_to_run]}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Device: {device}")
        logger.info("=" * 60)

        start_time = time.time()

        df, splits, data_processor = prepare_data(args, device)

        fold_results = []

        for fold in folds_to_run:
            logger.info(f"=== Fold {fold+1}/{args.n_folds} ===")
            try:
                fold_models = train_one_fold(args, device, df, splits, fold, model_config)
                ensemble, val_features, val_labels = create_ensemble_for_fold(
                    args, device, fold_models, df, splits, fold
                )
                metrics = evaluate_fold(ensemble, val_features, val_labels, fold, output_dir=args.output_dir)
                fold_results.append(metrics)
            except Exception as e:
                logger.error(f"Fold {fold+1} failed: {e}", exc_info=True)
                continue

        # Merge this run's fold results into any existing evaluation_results.json
        # (see merge_fold_results docstring) rather than overwriting it — this is
        # what makes it safe to resume a crashed multi-fold run with just the
        # remaining folds, e.g. --fold 3-4, without losing folds 0-2's scores.
        if fold_results:
            all_fold_results, avg_metrics = merge_fold_results(args.output_dir, fold_results)

            logger.info("Average Results Across All Folds Recorded So Far:")
            for metric_name, value in avg_metrics.items():
                if not metric_name.endswith('_std'):
                    std_value = avg_metrics.get(f'{metric_name}_std', 0)
                    logger.info(f"  {metric_name}: {value:.4f} \u00b1 {std_value:.4f}")

            results_path = os.path.join(args.output_dir, 'results', 'evaluation_results.json')
            logger.info(f"Evaluation results saved to {results_path} "
                        f"({len(all_fold_results)} fold(s) recorded total)")
        else:
            logger.warning("No folds completed successfully — no results to report.")

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training completed successfully in {format_time(total_time)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()