"""
Feature extraction using EfficientNetB0 backbone.
Extracts features for ensemble learning with SVM, RF, and KNN.
"""
import torch
import torch.nn as nn
from torchvision import models
import timm
import numpy as np
import logging
from typing import Tuple, Optional
import yaml
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml")

class EfficientNetB0FeatureExtractor(nn.Module):
    """
    EfficientNetB0 backbone for feature extraction.
    Outputs features before the final classification layer.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super(EfficientNetB0FeatureExtractor, self).__init__()

        # Load EfficientNetB0 from timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        logger.info(f"EfficientNetB0 feature extractor initialized with {self.feature_dim} features")

    def forward(self, x):
        """
        Extract features from input images.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        features = self.backbone.forward_features(x)
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        return flattened

def extract_features(model, dataloader, device):
    """
    Extract features from a dataset using the feature extractor.

    Args:
        model: EfficientNetB0FeatureExtractor instance
        dataloader: PyTorch DataLoader
        device: Computing device (cuda/cpu)

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            batch_features = model(data)
            features_list.append(batch_features.cpu().numpy())
            labels_list.append(target.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    logger.info(f"Extracted features shape: {features.shape}")
    return features, labels

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNetB0 with classification head for end-to-end training.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout_rate: float = 0.3):
        super(EfficientNetB0Classifier, self).__init__()

        # Load EfficientNetB0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Optionally freeze backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        logger.info(f"EfficientNetB0 classifier initialized")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone.forward_features(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        output = self.classifier(flattened)
        if self.classifier[-1].out_features == 1:
            output = output.squeeze(1)
        return output

class EffNetSklearnAdapter:
    """
    Wraps a trained EfficientNetB0Classifier so it can sit inside
    WeightedEnsembleClassifier alongside the sklearn-style models — it takes
    raw image tensors (not extracted features) and returns an (n, 2)
    probability matrix like a sklearn classifier's predict_proba does.
    """

    def __init__(self, torch_model: 'EfficientNetB0Classifier', device):
        self.model = torch_model.to(device).eval()
        self.device = device
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        """
        Args:
            X: batch of image tensors, shape (n_samples, 3, H, W)

        Returns:
            Probability estimates of shape (n_samples, 2): [P(real), P(fake)]
        """
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            probs_fake = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs_fake, probs_fake])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_model_config(config_path: str = DEFAULT_CONFIG_PATH):
    """Load model configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('model', {})
    else:
        # Default configuration
        return {
            'efficientnetb0': {
                'pretrained': True,
                'num_classes': 1,
                'dropout_rate': 0.3,
                'freeze_base': True,
                'unfreeze_from_layer': -30
            }
        }