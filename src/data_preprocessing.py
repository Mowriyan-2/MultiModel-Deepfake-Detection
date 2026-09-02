"""
Data preprocessing utilities for deepfake detection.
Handles loading, preprocessing, and augmentation of facial images.
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "model_config.yaml")

class DeepfakeDatasetProcessor:
    """Process and prepare deepfake datasets for training."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        import yaml
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
            self.config = {
                'preprocessing': {
                    'image_size': [224, 224],
                    'normalize_mean': [0.485, 0.456, 0.406],
                    'normalize_std': [0.229, 0.224, 0.225],
                }
            }

        self.image_size = tuple(self.config['preprocessing']['image_size'])
        self.normalize_mean = self.config['preprocessing']['normalize_mean']
        self.normalize_std = self.config['preprocessing']['normalize_std']

        # Define transforms
        self.train_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=10, p=0.5),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2()
        ])

    def load_faceforensics_data(self, data_path, compression='c23'):
        """
        Load FaceForensics++ dataset.

        Args:
            data_path: Path to FaceForensics++ dataset
            compression: Compression level ('c23' or 'c40')

        Returns:
            DataFrame with image paths, labels, and a video_id column
            (video_id is used downstream to split at the video level instead
            of the frame level — see create_stratified_splits).
        """
        data = []

        # Real videos
        real_path = os.path.join(data_path, 'original_sequences', 'youtube', compression, 'frames')
        if os.path.exists(real_path):
            for video_folder in os.listdir(real_path):
                video_path = os.path.join(real_path, video_folder)
                if os.path.isdir(video_path):
                    for frame_file in os.listdir(video_path):
                        if frame_file.endswith(('.png', '.jpg', '.jpeg')):
                            data.append({
                                'image_path': os.path.join(video_path, frame_file),
                                'label': 0,  # Real
                                'dataset': 'faceforensics',
                                'compression': compression,
                                # Prefixed with 'real' so a real video's ID can never
                                # collide with a fake video that happens to share a
                                # source-video filename under a different method.
                                'video_id': f"real_{video_folder}"
                            })

        # Fake videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
        methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for method in methods:
            fake_path = os.path.join(data_path, 'manipulated_sequences', method, compression, 'frames')
            if os.path.exists(fake_path):
                for video_folder in os.listdir(fake_path):
                    video_path = os.path.join(fake_path, video_folder)
                    if os.path.isdir(video_path):
                        for frame_file in os.listdir(video_path):
                            if frame_file.endswith(('.png', '.jpg', '.jpeg')):
                                data.append({
                                    'image_path': os.path.join(video_path, frame_file),
                                    'label': 1,  # Fake
                                    'dataset': 'faceforensics',
                                    'compression': compression,
                                    'method': method,
                                    'video_id': f"{method}_{video_folder}"
                                })

        df = pd.DataFrame(data)
        logger.info(f"Loaded FaceForensics++ {compression}: {len(df)} samples "
                   f"({len(df[df.label==0])} real, {len(df[df.label==1])} fake)")
        return df

    def load_celeb_df_data(self, data_path):
        """
        Load Celeb-DF dataset.

        Args:
            data_path: Path to Celeb-DF dataset

        Returns:
            DataFrame with image paths, labels, and a video_id column
            (see load_faceforensics_data docstring for why this matters).
        """
        data = []

        # Real videos
        real_path = os.path.join(data_path, 'Celeb-real')
        if os.path.exists(real_path):
            for video_folder in os.listdir(real_path):
                video_path = os.path.join(real_path, video_folder)
                if os.path.isdir(video_path):
                    for frame_file in os.listdir(video_path):
                        if frame_file.endswith(('.png', '.jpg', '.jpeg')):
                            data.append({
                                'image_path': os.path.join(video_path, frame_file),
                                'label': 0,  # Real
                                'dataset': 'celebdf',
                                'video_id': f"real_{video_folder}"
                            })

        # Fake videos
        fake_path = os.path.join(data_path, 'Celeb-synthesis')
        if os.path.exists(fake_path):
            for video_folder in os.listdir(fake_path):
                video_path = os.path.join(fake_path, video_folder)
                if os.path.isdir(video_path):
                    for frame_file in os.listdir(video_path):
                        if frame_file.endswith(('.png', '.jpg', '.jpeg')):
                            data.append({
                                'image_path': os.path.join(video_path, frame_file),
                                'label': 1,  # Fake
                                'dataset': 'celebdf',
                                'video_id': f"fake_{video_folder}"
                            })

        df = pd.DataFrame(data)
        logger.info(f"Loaded Celeb-DF: {len(df)} samples "
                   f"({len(df[df.label==0])} real, {len(df[df.label==1])} fake)")
        return df

    def create_stratified_splits(self, df, n_folds=5, random_state=42):
        """
        Create stratified k-fold splits, grouped by video.

        BUG FIX: the previous implementation ran StratifiedKFold directly on
        the frame-level dataframe, which let frames from the same video land
        in both the train and validation split of a fold. Since frames from
        one video are near-duplicates, that let the model "recognize" videos
        it had already partially seen, inflating validation accuracy. This
        version uses StratifiedGroupKFold with video_id as the group, so every
        frame of a given video is guaranteed to land entirely in either train
        or val for a given fold, never split across both.

        Args:
            df: DataFrame with image paths, labels, and a 'video_id' column
            n_folds: Number of folds
            random_state: Random seed

        Returns:
            List of (train_indices, val_indices) tuples
        """
        if 'video_id' not in df.columns:
            raise ValueError(
                "DataFrame is missing a 'video_id' column — group-aware "
                "splitting requires it. If you're loading data through a "
                "custom loader, add a 'video_id' column before calling this."
            )

        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = []

        for train_idx, val_idx in sgkf.split(df['image_path'], df['label'], groups=df['video_id']):
            splits.append((train_idx, val_idx))

        return splits

    def save_splits(self, splits, df, output_dir):
        """
        Save train/validation splits to disk.

        Args:
            splits: List of (train_indices, val_indices) tuples
            df: Original DataFrame
            output_dir: Directory to save splits
        """
        os.makedirs(output_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_df.to_csv(os.path.join(output_dir, f'fold_{fold}_train.csv'), index=False)
            val_df.to_csv(os.path.join(output_dir, f'fold_{fold}_val.csv'), index=False)

            logger.info(f"Fold {fold}: {len(train_df)} train, {len(val_df)} val samples")


class DeepfakeDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for deepfake detection."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load image
        image = Image.open(row['image_path']).convert('RGB')
        image = np.array(image)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)

        label = torch.tensor(row['label'], dtype=torch.float32)

        return image, label