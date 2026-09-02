"""
Training pipeline for deepfake detection models.
Includes early stopping, learning rate scheduling, and model checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import time
from typing import Tuple, Optional, Dict, Any
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml")

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Validation loss
            model: PyTorch model

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True

        return False

class LearningRateScheduler:
    """
    Learning rate scheduler with exponential decay.
    """

    def __init__(self, optimizer: optim.Optimizer, factor: float = 0.5, patience: int = 5, min_lr: float = 1e-7):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            factor: Factor to reduce LR by
            patience: Number of epochs to wait before reducing LR
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss: float):
        """
        Update learning rate based on validation loss.

        Args:
            val_loss: Validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                if old_lr != new_lr:
                    logger.info(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            self.counter = 0

class ModelTrainer:
    """
    Main training class for deepfake detection models.
    """

    def __init__(self, model: nn.Module, device: torch.device = None,
                 config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize the model trainer.

        Args:
            model: PyTorch model to train
            device: Computing device (cuda/cpu)
            config_path: Path to configuration file
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Load training configuration
        self._load_config(config_path)

        # Initialize training components
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping_patience'],
            min_delta=self.config['training']['early_stopping_min_delta']
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            factor=self.config['training']['lr_scheduler_factor'],
            patience=self.config['training']['lr_scheduler_patience'],
            min_lr=self.config['training']['min_lr']
        )

        self.criterion = nn.BCEWithLogitsLoss() if self.config['model']['efficientnetb0']['num_classes'] == 1 else nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # BUG FIX: previously there was no notion of a "best" checkpoint at
        # all — only periodic epoch_N snapshots (every 10 epochs) were saved,
        # and if early stopping never triggered, fit() returned whatever the
        # very last epoch left behind, even if an earlier epoch scored better.
        self.best_val_acc = -1.0
        self.best_val_loss = None
        self.best_state_dict = None

        logger.info(f"ModelTrainer initialized on {self.device}")

    def _load_config(self, config_path: str):
        """Load training configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Keep the nested structure — __init__ reads self.config['training'][...]
            # and self.config['model'][...], so flattening here would delete those keys.
            self.config = {
                'training': config.get('training', {}),
                'model': config.get('model', {}),
            }
            # YAML 1.1 (PyYAML's default loader) only recognizes scientific notation
            # with a decimal point (e.g. "1.0e-4"); bare "1e-4" parses as a string.
            # Coerce the numeric training hyperparameters defensively either way.
            for key in ('learning_rate', 'weight_decay', 'lr_scheduler_factor', 'min_lr'):
                if key in self.config['training']:
                    self.config['training'][key] = float(self.config['training'][key])

        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
            self.config = {
                'training': {
                    'learning_rate': 0.001,
                    'weight_decay': 1e-4,
                    'early_stopping_patience': 10,
                    'early_stopping_min_delta': 0.001,
                    'lr_scheduler_factor': 0.5,
                    'lr_scheduler_patience': 5,
                    'min_lr': 1e-7,
                    'epochs': 50,
                    'batch_size': 32
                },
                'model': {
                    'efficientnetb0': {
                        'num_classes': 1
                    }
                }
            }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            if output.dim() > 1 and output.shape[1] == 1:
                output = output.squeeze(1)

            # Calculate loss
            loss = self.criterion(output, target.float())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * data.size(0)

            # Calculate accuracy
            with torch.no_grad():
                pred = torch.sigmoid(output) >= 0.5
                correct_predictions += pred.eq(target.float()).sum().item()
                total_samples += data.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                if output.dim() > 1 and output.shape[1] == 1:
                    output = output.squeeze(1)

                # Calculate loss
                loss = self.criterion(output, target.float())

                # Statistics
                running_loss += loss.item() * data.size(0)

                # Calculate accuracy
                pred = torch.sigmoid(output) >= 0.5
                correct_predictions += pred.eq(target.float()).sum().item()
                total_samples += data.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None,
            epochs: int = None, save_dir: str = None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train (if None, uses config)
            save_dir: Directory to save checkpoints (optional)

        Returns:
            Dictionary containing training history. 'best_val_acc' is the
            best validation accuracy seen across all epochs this call — reuse
            this instead of re-running inference elsewhere (e.g. for ensemble
            weighting) when you need this model's validation performance.
        """
        if epochs is None:
            epochs = self.config['training']['epochs']

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

                # BUG FIX: track + persist the best-val-accuracy checkpoint.
                # Previously only periodic epoch_N snapshots existed and the
                # function always returned whatever the last epoch left
                # behind, which can be worse than an earlier epoch (this is
                # exactly what happened in the run that prompted this fix:
                # epoch 13 scored higher than epoch 15).
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    if save_dir is not None:
                        best_path = os.path.join(save_dir, "best_model.pth")
                        torch.save(self.best_state_dict, best_path)
                        logger.info(f"New best model (Val Acc: {val_acc:.4f}) saved to {best_path}")

                # Check early stopping
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                # Update learning rate
                self.lr_scheduler(val_loss)

                # Log progress
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save periodic checkpoint (separate from the best-model
                # tracking above — this is just a resumability snapshot)
                if save_dir is not None and (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
            else:
                # No validation
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # BUG FIX: restore the best-val-accuracy weights before returning,
        # regardless of whether early stopping triggered. EarlyStopping's own
        # restore_best_weights only fires on the early-stop path (and is
        # keyed off val_loss, which need not agree with val_acc); this makes
        # sure fit() always hands back its best model by validation accuracy.
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info(f"Restored best model weights (Val Acc: {self.best_val_acc:.4f})")

        # Return training history
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses if val_loader is not None else [],
            'val_accs': self.val_accs if val_loader is not None else [],
            'total_time': total_time,
            'epochs_trained': len(self.train_losses),
            'best_val_acc': self.best_val_acc if val_loader is not None else None,
            'best_val_loss': self.best_val_loss if val_loader is not None else None,
        }

        return history

def train_efficientnetb0(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                        epochs: int = 50, save_dir: str = None,
                        config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Convenience function to train an EfficientNetB0 model.

    Args:
        model: EfficientNetB0 model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        save_dir: Directory to save checkpoints
        config_path: Path to configuration file

    Returns:
        Training history dictionary
    """
    trainer = ModelTrainer(model, config_path=config_path)
    return trainer.fit(train_loader, val_loader, epochs, save_dir)