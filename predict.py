#!/usr/bin/env python3
"""
Prediction script for Multi-model-deep-fake-detection.
Loads trained models and runs inference on new images.
"""
import argparse
import os
import sys
import logging
import numpy as np
import torch
import cv2
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
from src.inference.batch_inference import BatchInferenceEngine
from src.inference.confidence_calibration import ConfidenceCalibrator
from src.visualization.gradcam import generate_gradcam_visualization, save_gradcam_results
from src.utils import (setup_logging, get_device, set_seed, create_directories,
                      load_config, get_system_info, log_system_info, format_time)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "model_config.yaml")


def load_models(args, device):
    """
    Load trained models for inference.

    BUG FIX (path layout mismatch): train.py saves models nested by fold —
    model_dir/svm/fold_N/svm_model.pkl, model_dir/efficientnetb0/fold_N/best_model.pth,
    model_dir/ensemble/fold_N/ensemble_model.pkl. This function previously
    looked for flat paths like model_dir/svm/svm_model.pkl and
    model_dir/efficientnetb0/model_best.pth — a filename that was never even
    produced by trainer.py (only periodic checkpoint_epoch_N.pth snapshots
    existed before the best-checkpoint fix). Nothing trained by train.py
    could ever be loaded through this path. Fixed to use the same
    fold_{args.fold}/ layout and the best_model.pth filename trainer.py now
    actually writes.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading models from {args.model_dir} (fold {args.fold})")

    model_dir = args.model_dir
    fold_tag = f"fold_{args.fold}"

    # Load EfficientNetB0 classifier
    efficientnet_model = EfficientNetB0Classifier(
        pretrained=False,
        num_classes=1
    ).to(device)

    effnet_dir = os.path.join(model_dir, 'efficientnetb0', fold_tag)
    best_path = os.path.join(effnet_dir, 'best_model.pth')
    if os.path.exists(best_path):
        state_dict = torch.load(best_path, map_location=device)
        efficientnet_model.load_state_dict(state_dict)
        logger.info(f"EfficientNetB0 best-checkpoint loaded from {best_path}")
    elif os.path.isdir(effnet_dir):
        # Fall back to the most recent periodic checkpoint if no best_model.pth
        # exists yet (e.g. models trained before the best-checkpoint fix).
        checkpoints = sorted(
            [f for f in os.listdir(effnet_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')],
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )
        if checkpoints:
            checkpoint_path = os.path.join(effnet_dir, checkpoints[-1])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"EfficientNetB0 periodic checkpoint loaded from {checkpoint_path} "
                        f"(no best_model.pth found — this model may not be from the best epoch)")
        else:
            logger.warning(f"No EfficientNetB0 checkpoint found in {effnet_dir}, using untrained ImageNet weights")
    else:
        logger.warning(f"EfficientNetB0 checkpoint directory not found: {effnet_dir}, using untrained ImageNet weights")

    # Load SVM, Random Forest, KNN — all under their own fold_N subfolder
    svm_model = SVMClassifier(args.model_config)
    svm_path = os.path.join(model_dir, 'svm', fold_tag, 'svm_model.pkl')
    if os.path.exists(svm_path):
        svm_model.load_model(svm_path)
        logger.info(f"SVM model loaded from {svm_path}")
    else:
        logger.warning(f"SVM model not found at {svm_path}")

    rf_model = RandomForestClassifier(args.model_config)
    rf_path = os.path.join(model_dir, 'random_forest', fold_tag, 'rf_model.pkl')
    if os.path.exists(rf_path):
        rf_model.load_model(rf_path)
        logger.info(f"Random Forest model loaded from {rf_path}")
    else:
        logger.warning(f"Random Forest model not found at {rf_path}")

    knn_model = KNNClassifier(args.model_config)
    knn_path = os.path.join(model_dir, 'knn', fold_tag, 'knn_model.pkl')
    if os.path.exists(knn_path):
        knn_model.load_model(knn_path)
        logger.info(f"KNN model loaded from {knn_path}")
    else:
        logger.warning(f"KNN model not found at {knn_path}")

    # The raw EfficientNetB0Classifier is a plain nn.Module with no
    # predict_proba, and it expects raw images rather than the 1280-dim
    # features SVM/RF/KNN use — wrap it in the adapter before adding it to
    # the ensemble. Also build a feature extractor sharing its backbone
    # weights for the SVM/RF/KNN feature-extraction path.
    cnn_adapter = EffNetSklearnAdapter(efficientnet_model, device)
    feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
    feature_extractor.backbone.load_state_dict(efficientnet_model.backbone.state_dict())
    feature_extractor.to(device).eval()

    ensemble = WeightedEnsembleClassifier(args.model_config)
    ensemble.add_model('efficientnetb0', cnn_adapter)
    ensemble.add_model('svm', svm_model)
    ensemble.add_model('random_forest', rf_model)
    ensemble.add_model('knn', knn_model)

    ensemble_path = os.path.join(model_dir, 'ensemble', fold_tag, 'ensemble_model.pkl')
    if os.path.exists(ensemble_path):
        # BUG FIX: load_ensemble() never restored the efficientnetb0 adapter
        # on its own (it only reconstructs svm/random_forest/knn from their
        # own pickles) — pass cnn_adapter explicitly so it gets re-attached
        # instead of silently vanishing from the loaded ensemble.
        ensemble.load_ensemble(ensemble_path, cnn_adapter=cnn_adapter)
        logger.info(f"Ensemble model loaded from {ensemble_path}")
    else:
        logger.info("No saved ensemble found — setting up ensemble with just-loaded individual models")
        dummy_features = np.zeros((1, 1280))  # EfficientNetB0 feature size
        dummy_labels = np.array([0])
        ensemble.fit(dummy_features, dummy_labels)

    return ensemble, feature_extractor, device


def predict_single_image(args, ensemble, feature_extractor, device):
    """Make prediction on a single image."""
    logger = logging.getLogger(__name__)
    logger.info(f"Making prediction on single image: {args.input}")

    from src.data_preprocessing import DeepfakeDataset
    from torch.utils.data import DataLoader
    from src.feature_extraction import extract_features
    import torchvision.transforms as transforms
    from PIL import Image

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    import pandas as pd
    df = pd.DataFrame({
        'image_path': [args.input],
        'label': [0]  # Dummy label
    })

    dataset = DeepfakeDataset(df, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Grab the (already-preprocessed) image batch once, and reuse it both for
    # feature extraction (SVM/RF/KNN) and as the raw input to the CNN adapter —
    # extract_features() consumes the dataloader, so pull the tensor out first.
    image_batch, _ = next(iter(dataloader))
    image_batch = image_batch.to(device)

    features, _ = extract_features(feature_extractor, dataloader, device)

    logger.info("Running ensemble prediction...")
    prediction = ensemble.predict(X_features=features, X_images=image_batch)[0]
    probabilities = ensemble.predict_proba(X_features=features, X_images=image_batch)[0]

    confidence = float(np.max(probabilities))
    label = 'FAKE' if prediction == 1 else 'REAL'

    logger.info(f"Prediction: {label} (confidence: {confidence:.4f})")
    logger.info(f"Probabilities: Real={probabilities[0]:.4f}, Fake={probabilities[1]:.4f}")

    if args.gradcam:
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            original_image = Image.open(args.input).convert('RGB')
            original_image_np = np.array(original_image)
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            # GradCAM needs the classifier (logit output), not the pooled feature
            # extractor — pull the wrapped torch model back out of the CNN adapter.
            classifier_model = ensemble.models['efficientnetb0'].model
            target_layer = "backbone.conv_head"  # timm efficientnet_b0's last conv layer
            cam, visualization = generate_gradcam_visualization(
                model=classifier_model,
                input_tensor=input_tensor,
                original_image=original_image_np,
                target_layer=target_layer,
                class_idx=int(prediction),
                alpha=0.4,
                colormap=cv2.COLORMAP_JET
            )

            output_dir = args.output or os.path.dirname(args.input) or '.'
            filename = os.path.splitext(os.path.basename(args.input))[0]
            save_gradcam_results(cam, visualization, original_image_np, output_dir, filename)
            logger.info(f"GradCAM visualization saved to {output_dir}")
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}", exc_info=True)

    return label, confidence


def predict_batch(args, ensemble, feature_extractor, device):
    """Make predictions on a batch of images or directory."""
    logger = logging.getLogger(__name__)

    if os.path.isdir(args.input):
        logger.info(f"Processing directory: {args.input}")
        inference_engine = BatchInferenceEngine(
            ensemble_model=ensemble,
            feature_extractor=feature_extractor,
            device=device,
            config_path=args.model_config
        )

        inference_engine.ensemble_model = ensemble
        inference_engine.ensemble_model.is_fitted = True

        results = inference_engine.predict_directory(
            args.input,
            extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            return_features=False
        )

        if args.output:
            inference_engine.save_results(results, args.output)
            logger.info(f"Batch results saved to {args.output}")
        else:
            output_path = os.path.join(os.path.dirname(args.input) or '.', 'predictions.csv')
            inference_engine.save_results(results, output_path)
            logger.info(f"Batch results saved to {output_path}")

        return results
    else:
        logger.info(f"Processing {len(args.input)} images")
        results = []
        for image_path in args.input:
            if os.path.exists(image_path):
                prediction, confidence = predict_single_image(
                    type('Args', (object,), {'input': image_path, 'output': None, 'gradcam': False})(),
                    ensemble, feature_extractor, device
                )
                results.append({
                    'image_path': image_path,
                    'prediction': prediction,
                    'confidence': confidence
                })
            else:
                logger.warning(f"Image not found: {image_path}")
        return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Run inference with Multi-model Deepfake Detection System')

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image, directory, or list of image paths')
    parser.add_argument('--output', type=str,
                       help='Path to save results (CSV for batch, directory for single-image GradCAM)')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory containing trained models (default: ./models)')
    parser.add_argument('--model_config', type=str,
                       default=DEFAULT_CONFIG_PATH,
                       help='Path to model configuration YAML file')
    parser.add_argument('--fold', type=int, default=1,
                       help='Which fold\'s trained models to load, 1-indexed (default: 1). '
                            'train.py saves each fold\'s models under a fold_N subfolder — '
                            'pick whichever fold scored best in your evaluation_results.json, '
                            'or just use fold 1 if you only trained a single fold.')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate GradCAM visualization (single image only)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    setup_logging(log_level="DEBUG" if args.verbose else "INFO")
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    device = get_device()

    try:
        ensemble, feature_extractor, device = load_models(args, device)

        if os.path.isdir(args.input) or (hasattr(args.input, '__iter__') and not isinstance(args.input, str)):
            results = predict_batch(args, ensemble, feature_extractor, device)
        else:
            prediction, confidence = predict_single_image(args, ensemble, feature_extractor, device)
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {confidence:.4f}\n")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()