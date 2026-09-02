"""
GradCAM (Gradient-weighted Class Activation Mapping) visualization
for interpreting deepfake detection results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class GradCAM:
    """
    GradCAM class for generating class activation maps.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize GradCAM.

        Args:
            model: PyTorch model
            target_layer: Name of the target layer for GradCAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
            return output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return grad_input

        # Find the target layer
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        # BUG FIX: register_backward_hook is deprecated, and PyTorch's own
        # docs note it has real correctness issues for some autograd graph
        # shapes (grad_input/grad_output aren't always what you'd expect).
        # register_full_backward_hook is the recommended replacement, with
        # the same grad_output[0] semantics used above.
        target_module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Class Activation Map.

        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            class_idx: Class index for which to generate CAM (if None, uses predicted class)

        Returns:
            CAM heatmap of shape (H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # If class_idx is not provided, use the predicted class
        if output.dim() == 1:  # Squeezed binary output (batch,) from num_classes=1
            if class_idx is None:
                class_idx = 1 if torch.sigmoid(output).item() > 0.5 else 0
        elif output.shape[1] == 1:  # Binary classification with sigmoid, shape (batch, 1)
            if class_idx is None:
                class_idx = 1 if output.item() > 0.5 else 0
        else:  # Multi-class
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

        # Backward pass
        if output.dim() == 1:
            loss = output[0] if class_idx == 1 else (1 - output[0])
        elif output.shape[1] == 1:  # Binary classification
            loss = output[0, 0] if class_idx == 1 else (1 - output[0, 0])
        else:  # Multi-class
            loss = output[0, class_idx]

        loss.backward()

        # Generate CAM
        gradients = self.gradients  # Shape: (1, C, H, W)
        activations = self.activations  # Shape: (1, C, H, W)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)  # Shape: (1, H, W)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Convert to numpy
        cam = cam.squeeze().cpu().detach().numpy()

        return cam

    def generate_visualization(self,
                             original_image: Union[np.ndarray, str, Image.Image],
                             cam: np.ndarray,
                             alpha: float = 0.4,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Generate visualization by overlaying CAM on original image.

        Args:
            original_image: Original image (as numpy array, file path, or PIL Image)
            cam: Class activation map
            alpha: Transparency factor for CAM overlay
            colormap: OpenCV colormap for CAM

        Returns:
            Visualization image as numpy array
        """
        # Load and preprocess original image
        if isinstance(original_image, str):
            img = cv2.imread(original_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(original_image, Image.Image):
            img = np.array(original_image)
        else:
            img = original_image.copy()

        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Apply colormap to CAM
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

        # Overlay CAM on original image
        visualization = img * (1 - alpha) + cam_colored * alpha
        visualization = np.clip(visualization, 0, 255).astype(np.uint8)

        return visualization

def generate_gradcam_visualization(model: nn.Module,
                                 input_tensor: torch.Tensor,
                                 original_image: Union[np.ndarray, str, Image.Image],
                                 target_layer: str = "backbone.conv_head",
                                 class_idx: int = None,
                                 alpha: float = 0.4,
                                 colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate GradCAM visualization.

    Args:
        model: PyTorch model
        input_tensor: Input tensor of shape (1, 3, H, W)
        original_image: Original image for visualization
        target_layer: Target layer for GradCAM
        class_idx: Class index for visualization (if None, uses predicted class)
        alpha: Transparency factor
        colormap: OpenCV colormap

    Returns:
        Tuple of (CAM heatmap, visualization image)
    """
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, class_idx)
    visualization = gradcam.generate_visualization(original_image, cam, alpha, colormap)

    return cam, visualization

def save_gradcam_results(cam: np.ndarray,
                        visualization: np.ndarray,
                        original_image: Union[np.ndarray, str, Image.Image],
                        save_dir: str,
                        filename: str = "gradcam_result"):
    """
    Save GradCAM results to disk.

    Args:
        cam: CAM heatmap
        visualization: GradCAM visualization image
        original_image: Original image
        save_dir: Directory to save results
        filename: Base filename for saved files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Save original image
    if isinstance(original_image, str):
        orig_img = cv2.imread(original_image)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    elif isinstance(original_image, Image.Image):
        orig_img = np.array(original_image)
    else:
        orig_img = original_image.copy()

    cv2.imwrite(os.path.join(save_dir, f"{filename}_original.jpg"),
                cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))

    # Save CAM heatmap
    cam_normalized = (cam * 255).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, f"{filename}_cam.jpg"), cam_colored)

    # Save visualization
    cv2.imwrite(os.path.join(save_dir, f"{filename}_visualization.jpg"),
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    logger.info(f"GradCAM results saved to {save_dir}")