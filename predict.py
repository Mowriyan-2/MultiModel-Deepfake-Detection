#!/usr/bin/env python3
"""
Prediction script for Multi-model-deep-fake-detection.
Handles loading trained models and making predictions on new data.
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

def setup_environment(args):
    """Setup logging, device, seeds, and directories."""
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level, log_file=args.log_file)

    # Get device
    device = get_device()

    return device

def load_models(args, device):
    """Load trained models from disk."""
    logger = logging.getLogger(__name__)
    logger.info("Loading trained models...")

    # Initialize individual models
    efficientnet_model = EfficientNetB0Classifier(
        pretrained=False,  # We'll load our trained weights
        num_classes=1
    ).to(device)

    svm_model = SVMClassifier(args.model_config)
    rf_model = RandomForestClassifier(args.model_config)
    knn_model = KNNClassifier(args.model_config)

    # Load models from specified directory
    model_dir = args.model_dir

    # Load EfficientNetB0
    effnet_path = os.path.join(model_dir, 'efficientnetb0', 'model_best.pth')
    if os.path.exists(effnet_path):
        efficientnet_model.load_state_dict(torch.load(effnet_path, map_location=device))
        efficientnet_model.eval()
        logger.info(f"EfficientNetB0 model loaded from {effnet_path}")
    else:
        # Try to find any checkpoint
        effnet_dir = os.path.join(model_dir, 'efficientnetb0')
        if os.path.exists(effnet_dir):
            checkpoints = [f for f in os.listdir(effnet_dir) if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                effnet_path = os.path.join(effnet_dir, latest_checkpoint)
                efficientnet_model.load_state_dict(torch.load(effnet_path, map_location=device))
                efficientnet_model.eval()
                logger.info(f"EfficientNetB0 model loaded from {effnet_path}")
            else:
                logger.warning(f"No EfficientNetB0 checkpoints found in {effnet_dir}")
                logger.info("Using ImageNet pretrained weights")
        else:
            logger.warning(f"EfficientNetB0 directory not found: {effnet_dir}")
            logger.info("Using ImageNet pretrained weights")

    # Load traditional ML models
    try:
        svm_path = os.path.join(model_dir, 'svm', 'svm_model.pkl')
        if os.path.exists(svm_path):
            svm_model.load_model(svm_path)
            logger.info(f"SVM model loaded from {svm_path}")
        else:
            logger.warning(f"SVM model not found at {svm_path}")
    except Exception as e:
        logger.warning(f"Failed to load SVM model: {e}")

    try:
        rf_path = os.path.join(model_dir, 'random_forest', 'rf_model.pkl')
        if os.path.exists(rf_path):
            rf_model.load_model(rf_path)
            logger.info(f"Random Forest model loaded from {rf_path}")
        else:
            logger.warning(f"Random Forest model not found at {rf_path}")
    except Exception as e:
        logger.warning(f"Failed to load Random Forest model: {e}")

    try:
        knn_path = os.path.join(model_dir, 'knn', 'knn_model.pkl')
        if os.path.exists(knn_path):
            knn_model.load_model(knn_path)
            logger.info(f"KNN model loaded from {knn_path}")
        else:
            logger.warning(f"KNN model not found at {knn_path}")
    except Exception as e:
        logger.warning(f"Failed to load KNN model: {e}")

    # Create and load ensemble
    # The raw EfficientNetB0Classifier is a plain nn.Module with no predict_proba,
    # so wrap it in the sklearn-compatible adapter before adding it to the ensemble.
    # SVM/RF/KNN were trained on pooled 1280-dim features, so give them their own
    # feature extractor sharing the classifier's backbone weights.
    cnn_adapter = EffNetSklearnAdapter(efficientnet_model, device)
    feature_extractor = EfficientNetB0FeatureExtractor(pretrained=False)
    feature_extractor.backbone.load_state_dict(efficientnet_model.backbone.state_dict())
    feature_extractor.to(device).eval()

    ensemble = WeightedEnsembleClassifier(args.model_config)
    ensemble.add_model('efficientnetb0', cnn_adapter)
    ensemble.add_model('svm', svm_model)
    ensemble.add_model('random_forest', rf_model)
    ensemble.add_model('knn', knn_model)

    ensemble_path = os.path.join(model_dir, 'ensemble', 'ensemble_model.pkl')
    if os.path.exists(ensemble_path):
        ensemble.load_ensemble(ensemble_path)
        logger.info(f"Ensemble model loaded from {ensemble_path}")
    else:
        logger.info("Setting up ensemble with loaded models")
        # Initialize ensemble (models already added)
        # Need some dummy data to fit the ensemble structure
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

    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create dataset for single image
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

    # Extract pooled 1280-dim features for the sklearn-style models
    features, _ = extract_features(feature_extractor, dataloader, device)

    # Make prediction — route features to SVM/RF/KNN and the raw image batch to
    # the CNN adapter (the two require different input shapes, see ensemble_voting.py)
    logger.info("Running ensemble prediction...")
    prediction = ensemble.predict(X_features=features, X_images=image_batch)[0]
    probabilities = ensemble.predict_proba(X_features=features, X_images=image_batch)[0]

    # Get confidence (probability of predicted class)
    confidence = probabilities[prediction]

    # Format output
    label_map = {0: "Real", 1: "Fake"}
    prediction_label = label_map[prediction]

    logger.info(f"Prediction: {prediction_label}")
    logger.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    logger.info(f"Probabilities - Real: {probabilities[0]:.4f}, Fake: {probabilities[1]:.4f}")

    # Save results if requested
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        import json
        results = {
            'input_image': args.input,
            'prediction': prediction_label,
            'confidence': float(confidence),
            'probability_real': float(probabilities[0]),
            'probability_fake': float(probabilities[1])
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Generate GradCAM visualization if requested
    if args.gradcam:
        logger.info("Generating GradCAM visualization...")
        try:
            # Load and preprocess image for GradCAM
            original_image = Image.open(args.input).convert('RGB')
            original_image_np = np.array(original_image)

            # Preprocess for model input
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(original_image).unsqueeze(0).to(device)

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

            # Save visualization
            output_dir = os.path.dirname(args.output) if args.output else "./"
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            viz_path = os.path.join(output_dir, f"{base_name}_gradcam.jpg")
            save_gradcam_results(cam, visualization, original_image_np, output_dir, base_name)
            logger.info(f"GradCAM visualization saved to {viz_path}")

        except Exception as e:
            logger.warning(f"Could not generate GradCAM visualization: {e}")

    return prediction_label, confidence

def predict_batch(args, ensemble, feature_extractor, device):
    """Make predictions on a batch of images or directory."""
    logger = logging.getLogger(__name__)

    if os.path.isdir(args.input):
        logger.info(f"Processing directory: {args.input}")
        # Use batch inference engine
        inference_engine = BatchInferenceEngine(
            ensemble_model=ensemble,
            feature_extractor=feature_extractor,
            device=device,
            config_path=args.model_config
        )

        # Load models into inference engine
        inference_engine.ensemble_model = ensemble
        inference_engine.ensemble_model.is_fitted = True

        results = inference_engine.predict_directory(
            args.input,
            extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            return_features=False
        )

        # Save results
        if args.output:
            inference_engine.save_results(results, args.output)
            logger.info(f"Batch results saved to {args.output}")
        else:
            # Default output path
            output_path = os.path.join(os.path.dirname(args.input) or '.', 'predictions.csv')
            inference_engine.save_results(results, output_path)
            logger.info(f"Batch results saved to {output_path}")

        return results
    else:
        # Treat as list of image files
        logger.info(f"Processing {len(args.input)} images")
        # For simplicity, we'll process them individually
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
    parser = argparse.ArgumentParser(description='Predict with Multi-model Deepfake Detection System')

    # Input arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file, directory, or list of image files')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory containing trained models (default: ./models)')
    parser.add_argument('--model_config', type=str,
                       default=DEFAULT_CONFIG_PATH,
                       help='Path to model configuration YAML file')

    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Output file for predictions (JSON for single image, CSV for batch)')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate GradCAM visualization (for single image only)')

    # Processing arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log_file', type=str,
                       help='Path to log file (optional)')

    args = parser.parse_args()

    try:
        # Setup environment
        device = setup_environment(args)

        logger = logging.getLogger(__name__)
        logger.info("="*60)
        logger.info("Multi-model Deepfake Detection Prediction Started")
        logger.info("="*60)
        logger.info(f"Input: {args.input}")
        logger.info(f"Model directory: {args.model_dir}")
        logger.info(f"Generate GradCAM: {args.gradcam}")
        logger.info("="*60)

        start_time = time.time()

        # Load models
        ensemble, feature_extractor, device = load_models(args, device)

        # Make predictions
        if os.path.isdir(args.input) or (hasattr(args.input, '__iter__') and not isinstance(args.input, str)):
            # Batch prediction
            results = predict_batch(args, ensemble, feature_extractor, device)
        else:
            # Single image prediction
            prediction, confidence = predict_single_image(args, ensemble, feature_extractor, device)

        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"Prediction completed in {format_time(total_time)}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()