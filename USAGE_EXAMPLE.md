# Usage Examples for Multi-model-deep-fake-detection

## 1. Training the Model

### Train on FaceForensics++ (C23 compression)
```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --compression c23 \\
    --epochs 50 \\
    --batch_size 32 \\
    --save_models
```

### Train on Celeb-DF
```bash
python train.py \\
    --dataset celebdf \\
    --data_path /path/to/celeb-df \\
    --epochs 50 \\
    --batch_size 32 \\
    --save_models
```

### Train with 5-fold Cross Validation (default)
```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --compression c23 \\
    --n_folds 5 \\
    --save_models
```

### Train a Specific Fold
```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --compression c23 \\
    --fold 0  # Train only fold 1 (0-indexed) \\
    --save_models
```

### Resume Training from Checkpoint
```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --compression c23 \\
    --resume_from ./models/efficientnetb0/fold_1/checkpoint_epoch_25.pth
```

## 2. Making Predictions

### Predict on a Single Image
```bash
python predict.py \\
    --input /path/to/test_image.jpg \\
    --model_dir ./models \\
    --output ./results/prediction.json \\
    --gradcam
```

### Predict on a Directory of Images
```bash
python predict.py \\
    --input /path/to/test_images/ \\
    --model_dir ./models \\
    --output ./results/batch_predictions.csv
```

### Predict on Multiple Specific Images
```bash
python predict.py \\
    --input /path/to/image1.jpg /path/to/image2.jpg /path/to/image3.jpg \\
    --model_dir ./models \\
    --output ./results/multiple_predictions.json
```

## 3. Evaluation Only (Using Pre-trained Models)

```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --compression c23 \\
    --evaluate_only \\
    --model_dir ./models
```

## 4. Verbose Logging and Debugging

```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --compression c23 \\
    --verbose \\
    --log_file ./logs/training.log
```

## 5. Using Custom Configurations

```bash
python train.py \\
    --dataset faceforensics \\
    --data_path /path/to/faceforensics++ \\
    --model_config ./configs/custom_model.yaml \\
    --training_config ./configs/custom_training.yaml
```

## Expected Output Structure

After training, your directory structure will look like:

```
Multi-model-deep-fake-detection/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── efficientnetb0/
│   │   ├── fold_1/
│   │   │   ├── model_best.pth
│   │   │   └── checkpoint_epoch_*.pth
│   │   ├── fold_2/
│   │   │   └── ...
│   ├── svm/
│   │   ├── fold_1/
│   │   │   └── svm_model.pkl
│   │   ├── fold_2/
│   │   │   └── ...
│   ├── random_forest/
│   │   ├── fold_1/
│   │   │   └── rf_model.pkl
│   │   ├── fold_2/
│   │   │   └── ...
│   ├── knn/
│   │   ├── fold_1/
│   │   │   └── knn_model.pkl
│   │   ├── fold_2/
│   │   │   └── ...
│   └── ensemble/
│       ├── fold_1/
│       │   └── ensemble_model.pkl
│       ├── fold_2/
│       │   └── ...
│       └── ensemble_model.pkl  # Combined ensemble
├── output/
│   ├── logs/
│   ├── plots/
│   └── results/
│       ├── evaluation_results.json
│       ├── training_history.png
│       └── confusion_matrix.png
├── src/
├── notebooks/
├── configs/
├── scripts/
├── train.py
├── predict.py
�└── README.md
```

## Sample Output

When running training, you should see output similar to:

```
============================================================
Multi-model Deepfake Detection Training Started
============================================================
Dataset: faceforensics
Data path: /path/to/faceforensics++
Compression: c23
Number of folds: 5
Training fold: All folds
Epochs: 50
Batch size: 32
Learning rate: 0.001
Seed: 42
Device: cuda
============================================================

System Information:
  platform: nt
  python_version: 3.9.16
  cpu_count: 16
  memory_total_gb: 32.00
  memory_available_gb: 18.50
  gpu_count: 1
  gpu_names: ['NVIDIA GeForce RTX 3080']
  gpu_memory_gb: [10.00]

======== Starting Fold 1/5 ========
Training EfficientNetB0...
Epoch 1/50 - Train Loss: 0.6523, Train Acc: 0.6123 - Val Loss: 0.5891, Val Acc: 0.6456 - Time: 12.34s
Epoch 2/50 - Train Loss: 0.4856, Train Acc: 0.7654 - Val Loss: 0.4567, Val Acc: 0.7890 - Time: 11.98s
...
Early stopping triggered at epoch 23 without improvement

Training SVM...
SVM training completed in 00:00:03

Training Random Forest...
Random Forest training completed in 00:00:15

Training KNN...
KNN training completed in 00:00:02

Fold 1 completed - Val Loss: 0.3456, Val Acc: 0.8456

======== Starting Fold 2/5 ========
...
```

When making predictions, you should see output similar to:

```
============================================================
Multi-model Deepfake Detection Prediction Started
============================================================
Input: /path/to/test_image.jpg
Model directory: ./models
Generate GradCAM: True
============================================================

Loading trained models...
������✓ EfficientNetB0 model loaded from ./models/efficientnetb0/fold_1/model_best.pth
������✓ SVM model loaded from ./models/svm/fold_1/svm_model.pkl
������✓ Random Forest model loaded from ./models/random_forest/fold_1/rf_model.pkl
������✓ KNN model loaded from ./models/knn/fold_1/knn_model.pkl
������ Ensemble model loaded from ./models/ensemble/ensemble_model.pkl

Making prediction on single image: /path/to/test_image.jpg
Running ensemble prediction...
Prediction: Fake
Confidence: 0.8473 (84.73%)
Probabilities - Real: 0.1527, Fake: 0.8473

Generating GradCAM visualization...
������ GradCAM visualization saved to ./results/test_image_gradcam.jpg

============================================================
Prediction completed in 00:00:05
============================================================
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: The system will automatically fall back to CPU if CUDA is not available
2. **Memory errors**: Try reducing the batch size (`--batch_size 16` or `--batch_size 8`)
3. **Missing dependencies**: Make sure all requirements are installed with `pip install -r requirements.txt`
4. **Dataset path errors**: Ensure the dataset path points to the correct directory structure
5. **Model loading errors**: Make sure you have trained models in the specified model directory

### Getting Help

```bash
python train.py --help
python predict.py --help
```

For additional support, please refer to the README.md file or contact the system administrator.