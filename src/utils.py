"""
Utility functions for deepfake detection project.
Common helper functions used across the project.
"""
import os
import torch
import numpy as np
import logging
import json
import yaml
from typing import Tuple, List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=numeric_level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=numeric_level, format=format_str)

def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, otherwise CPU).

    Returns:
        Torch device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")

    return device

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str,
                         metadata: Dict[str, Any] = None):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metadata': metadata or {}
    }

    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         filepath: str, device: torch.device = None) -> Tuple[int, float]:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (epoch, loss)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metadata = checkpoint.get('metadata', {})

    logger.info(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seeds set to {seed}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging.

    Returns:
        Dictionary containing system information
    """
    info = {
        'platform': os.name,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'cpu_count': os.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / 1e9,
        'memory_available_gb': psutil.virtual_memory().available / 1e9
    }

    # GPU information if available
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info['gpu_memory_gb'] = [torch.cuda.get_device_properties(i).total_memory / 1e9
                                for i in range(torch.cuda.device_count())]

    return info

def log_system_info():
    """Log system information."""
    info = get_system_info()
    logger.info("System Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

def create_directories(directory_paths: List[str]):
    """
    Create directories if they don't exist.

    Args:
        directory_paths: List of directory paths to create
    """
    for path in directory_paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML configuration file
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise

def plot_training_history(history: Dict[str, List[float]],
                         save_path: str = None,
                         figsize: Tuple[int, int] = (12, 4)):
    """
    Plot training history (loss and accuracy).

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (if None, displays plot)
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    axes[0].plot(history.get('train_losses', []), label='Train Loss', color='blue')
    if 'val_losses' in history and len(history['val_losses']) > 0:
        axes[0].plot(history['val_losses'], label='Val Loss', color='red')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history.get('train_accs', []), label='Train Acc', color='blue')
    if 'val_accs' in history and len(history['val_accs']) > 0:
        axes[1].plot(history['val_accs'], label='Val Acc', color='red')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None,
                         save_path: str = None, figsize: Tuple[int, int] = (8, 6)):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix numpy array
        class_names: List of class names
        save_path: Path to save the plot (if None, displays plot)
        figsize: Figure size tuple
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_prob is not None:
        try:
            # For binary classification, use probability of positive class
            if y_prob.ndim > 1 and y_prob.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = 0.0

    return metrics

def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.

    Returns:
        Dictionary containing memory usage information
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        'rss_mb': memory_info.rss / 1e6,  # Resident Set Size
        'vms_mb': memory_info.vms / 1e6,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def clear_gpu_memory():
    """
    Clear GPU memory cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")

class Timer:
    """
    Simple timer context manager for measuring execution time.
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and isinstance(self.start_time, torch.cuda.Event):
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = self.start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            elapsed_time = time.time() - self.start_time

        logger.info(f"{self.name} completed in {format_time(elapsed_time)} ({elapsed_time:.2f}s)")


# Import time at module level to avoid circular imports
import time