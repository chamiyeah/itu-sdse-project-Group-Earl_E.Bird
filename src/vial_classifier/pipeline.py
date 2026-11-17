import json
import os

import mlflow
import torch

from vial_classifier.data.dataset import (
    calculate_image_statistics,
    create_dataloaders,
    load_image_dataset,
    split_dataset,
)
from vial_classifier.drift.baseline import compute_baseline_statistics, log_baseline_to_mlflow
from vial_classifier.features.transforms import define_transforms
from vial_classifier.models.cnn import SimpleCNN, get_device
from vial_classifier.seed import set_seed
from vial_classifier.tracking import setup_mlflow_tracking
from vial_classifier.training.evaluate import evaluate_cnn_model
from vial_classifier.training.threshold import tune_threshold
from vial_classifier.training.train import train_cnn_model


def _print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def run_training(config):
    """Run the whole training job in the same order as the training notebook.

    This is the only function that reads Config: every module below receives
    plain values as parameters. The comments name the notebook cell of each step.
    """
    # --- cell 4: folders ---
    os.makedirs(config.artifact_dir, exist_ok=True)
    os.makedirs(config.mlflow_state_dir, exist_ok=True)
    os.makedirs(config.mlflow_artifact_location, exist_ok=True)
    os.makedirs(config.image_data_dir, exist_ok=True)
    print("Created artifacts directory")

    # --- cell 8: MLflow setup ---
    setup_mlflow_tracking(
        config.mlflow_state_dir,
        config.mlflow_tracking_uri,
        config.mlflow_artifact_location,
        config.experiment_name,
    )
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # --- cell 13: load the (image_path, label) pairs ---
    samples = load_image_dataset(config.image_data_dir, config.metadata_path)

    # --- cell 19: class names artifact ---
    class_names = {0: "good", 1: "defective"}
    with open(config.class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {config.class_names_path}")

    # --- cell 21: transforms ---
    train_transforms, val_transforms = define_transforms(
        config.image_size, config.transform_mean, config.transform_std
    )

    # --- cell 24: split, DataLoaders, image statistics ---
    # No seed is passed to split_dataset: the notebook uses its default (42) too.
    train_samples, val_samples, test_samples = split_dataset(samples)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, train_transforms, val_transforms, config.batch_size
    )

    # The statistics read the training loader (shuffle + augmentations), so they
    # draw random numbers: they must stay here, before the seed of cell 31.
    image_stats = calculate_image_statistics(train_loader)
    with open(config.image_statistics_path, "w") as f:
        json.dump(image_stats, f, indent=2)
    print(f"Image statistics saved to {config.image_statistics_path}")

    # --- cell 26: device (only once) ---
    device = get_device()

    # --- cell 31: training run ---
    _print_section_header("CNN MODEL TRAINING")

    set_seed(config.random_state)

    if mlflow.active_run():
        mlflow.end_run()
        print("Ended previous active MLflow run")

    # No try/finally, like the notebook: if a step below fails, the run stays open.
    mlflow.start_run(run_name="training_notebook")

    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("num_epochs", config.num_epochs)
    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("initial_threshold", config.initial_threshold)
    mlflow.log_param("num_classes", config.num_classes)
    mlflow.log_param("image_size", "64x64")
    mlflow.log_param("random_seed", config.random_state)

    # The notebook builds two throw-away models just to count the parameters.
    # Each one draws from the global torch generator, so both are kept.
    model_params = sum(p.numel() for p in SimpleCNN().parameters())
    trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", model_params)
    mlflow.log_param("trainable_parameters", trainable_params)

    cnn_model, training_history = train_cnn_model(
        train_loader,
        val_loader,
        device,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        random_seed=config.random_state,
        threshold=config.initial_threshold,
        early_stop_patience=config.early_stop_patience,
        scheduler_patience=config.scheduler_patience,
    )

    # Tune threshold on validation set
    optimal_threshold = tune_threshold(
        cnn_model, val_loader, device, initial_threshold=config.initial_threshold
    )

    # Evaluate on test set with optimal threshold.
    # The threshold is passed as it is (a np.float64), without conversions.
    test_results = evaluate_cnn_model(cnn_model, test_loader, device, threshold=optimal_threshold)

    # --- cell 33: drift baseline ---
    baseline_stats = compute_baseline_statistics(
        cnn_model, val_loader, device, threshold=optimal_threshold
    )

    # The file must be written before log_baseline_to_mlflow, which uploads it.
    with open(config.drift_baseline_stats_path, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Baseline statistics saved to {config.drift_baseline_stats_path}")

    log_baseline_to_mlflow(baseline_stats, config.drift_baseline_stats_path)

    # --- cell 37 (first part): model artifacts ---
    print("\nSaving model artifacts...")

    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'architecture': SimpleCNN.__name__,
        'input_shape': (3, 64, 64),
        'output_shape': (1,),
        'threshold': optimal_threshold,
        'num_classes': config.num_classes,
    }, config.pytorch_model_path)
    print(f"CNN model saved to {config.pytorch_model_path}")

    with open(config.model_results_path, "w") as results_file:
        json.dump(test_results, results_file, indent=2)
    print(f"Model results saved to {config.model_results_path}")

    mlflow.log_param("classification_threshold", optimal_threshold)

    # --- cell 40 (first line): close the run ---
    mlflow.end_run()

    return {
        "optimal_threshold": optimal_threshold,
        "test_results": test_results,
        "artifact_paths": {
            "image_statistics": config.image_statistics_path,
            "class_names": config.class_names_path,
            "drift_baseline_stats": config.drift_baseline_stats_path,
            "pytorch_model": config.pytorch_model_path,
            "model_results": config.model_results_path,
        },
    }
