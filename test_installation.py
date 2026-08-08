#!/usr/bin/env python3
"""
Test script to verify Multi-model-deep-fake-detection installation.
"""
import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Test src modules
        import src.data_preprocessing
        print("data_preprocessing imported successfully")

        import src.feature_extraction
        print("feature_extraction imported successfully")

        import src.models.svm_classifier
        print("svm_classifier imported successfully")

        import src.models.random_forest_classifier
        print("random_forest_classifier imported successfully")

        import src.models.knn_classifier
        print("knn_classifier imported successfully")

        import src.models.ensemble_voting
        print("ensemble_voting imported successfully")

        import src.visualization.gradcam
        print("gradcam imported successfully")

        import src.training.trainer
        print("trainer imported successfully")

        import src.training.cross_validation
        print("cross_validation imported successfully")

        import src.inference.batch_inference
        print("batch_inference imported successfully")

        import src.inference.confidence_calibration
        print("confidence_calibration imported successfully")

        import src.utils
        print("utils imported successfully")

        # Test main modules
        import train
        print("train imported successfully")

        import predict
        print("predict imported successfully")

        print("\nAll imports successful!")
        return True

    except Exception as e:
        print(f"\nImport failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directories():
    """Test that required directories exist or can be created."""
    print("\nTesting directory structure...")

    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'data/splits',
        'models',
        'models/efficientnetb0',
        'models/svm',
        'models/random_forest',
        'models/knn',
        'models/ensemble',
        'src',
        'src/data_preprocessing',
        'src/feature_extraction',
        'src/models',
        'src/visualization',
        'src/training',
        'src/inference',
        'notebooks',
        'configs',
        'scripts'
    ]

    base_path = os.path.dirname(os.path.abspath(__file__))
    all_good = True

    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if os.path.exists(full_path):
            print(f"{dir_path}")
        else:
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"{dir_path} (created)")
            except Exception as e:
                print(f"{dir_path} - Failed to create: {e}")
                all_good = False

    return all_good

def test_config_files():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")

    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, 'configs', 'model_config.yaml')

    if os.path.exists(config_path):
        print("configs/model_config.yaml")
        return True
    else:
        print("configs/model_config.yaml (missing)")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Multi-model-deep-fake-detection Installation Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_directories,
        test_config_files
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    if all(results):
        print("All tests passed! Installation is ready.")
        print("You can now use:")
        print("  python train.py --help  # See training options")
        print("  python predict.py --help  # See prediction options")
    else:
        print("Some tests failed. Please check the output above.")
        print("Make sure you are in the project directory and have all dependencies installed.")
    print("=" * 50)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())