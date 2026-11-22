import json
import os

import mlflow
import torch

from vial_classifier.data.dataset import create_dataloaders, load_image_dataset, split_dataset
from vial_classifier.drift.baseline import compute_baseline_statistics, log_baseline_to_mlflow
from vial_classifier.features.transforms import define_transforms
from vial_classifier.models.cnn import SimpleCNN, get_device
from vial_classifier.tracking import setup_mlflow_tracking
from vial_classifier.training.evaluate import evaluate_cnn_model
from vial_classifier.training.threshold import tune_threshold
from vial_classifier.training.train import train_cnn_model


def run_training(config):
    """Run the whole training job: data, training, threshold, evaluation, drift baseline.

    This is the only function that reads Config: every module below receives
    plain values as parameters. The ML logic comes from the training notebook
    (notebooks/01_training.ipynb); the comments name the cell each step comes from.
    """
    # The other folders are created where they are needed: the MLflow ones by
    # setup_mlflow_tracking and by MLflow itself, the image folder by `acquire`.
    os.makedirs(config.artifact_dir, exist_ok=True)

    # MLflow setup (nb01 c8)
    setup_mlflow_tracking(
        config.mlflow_state_dir,
        config.mlflow_tracking_uri,
        config.mlflow_artifact_location,
        config.experiment_name,
    )
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Load the (image_path, label) pairs (nb01 c13)
    samples = load_image_dataset(config.image_data_dir, config.metadata_path)

    # Class names artifact (nb01 c19)
    class_names = {0: "good", 1: "defective"}
    with open(config.class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {config.class_names_path}")

    # Transforms, split and DataLoaders (nb01 c21, c24)
    train_transforms, val_transforms = define_transforms(
        config.image_size, config.transform_mean, config.transform_std
    )
    train_samples, val_samples, test_samples = split_dataset(samples, random_seed=config.random_state)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples, val_samples, test_samples, train_transforms, val_transforms, config.batch_size
    )

    device = get_device()

    # Training run (nb01 c31, c33, c37). The `with` block closes the run at the
    # end, and marks it FAILED if any step inside raises an error.
    with mlflow.start_run(run_name="training"):
        # image_size goes to transforms.Resize, which reads it as (height, width)
        height, width = config.image_size
        mlflow.log_param("model_architecture", "SimpleCNN")
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("num_epochs", config.num_epochs)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("initial_threshold", config.initial_threshold)
        mlflow.log_param("num_classes", config.num_classes)
        mlflow.log_param("image_size", f"{width}x{height}")
        mlflow.log_param("random_seed", config.random_state)

        # train_cnn_model seeds the random generators itself before building the model
        cnn_model, _ = train_cnn_model(
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

        total_params = sum(p.numel() for p in cnn_model.parameters())
        trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)

        # Tune the threshold on the validation set. tune_threshold also logs it
        # as the "optimal_threshold" parameter.
        optimal_threshold = tune_threshold(
            cnn_model, val_loader, device, initial_threshold=config.initial_threshold
        )

        # The threshold stays a np.float64 here. The probabilities are float32:
        # with NumPy 2, comparing them to a np.float64 happens in float64 (like
        # in tune_threshold), but comparing them to a plain Python float happens
        # in float32, which can flip a prediction that sits right on the threshold.
        test_results = evaluate_cnn_model(cnn_model, test_loader, device, threshold=optimal_threshold)

        # Drift baseline (nb01 c33)
        baseline_stats = compute_baseline_statistics(
            cnn_model, val_loader, device, threshold=optimal_threshold
        )
        # The file must be written before log_baseline_to_mlflow, which uploads it.
        with open(config.drift_baseline_stats_path, "w") as f:
            json.dump(baseline_stats, f, indent=2)
        print(f"Baseline statistics saved to {config.drift_baseline_stats_path}")
        log_baseline_to_mlflow(baseline_stats, config.drift_baseline_stats_path)

        # Model artifacts (nb01 c37)
        torch.save({
            'model_state_dict': cnn_model.state_dict(),
            'architecture': SimpleCNN.__name__,
            'input_shape': (3, *config.image_size),
            'output_shape': (1,),
            # Saved as a plain float (it is only stored, not compared to anything),
            # so the file holds no NumPy objects.
            'threshold': float(optimal_threshold),
            'num_classes': config.num_classes,
        }, config.pytorch_model_path)
        print(f"CNN model saved to {config.pytorch_model_path}")

        with open(config.model_results_path, "w") as results_file:
            json.dump(test_results, results_file, indent=2)
        print(f"Model results saved to {config.model_results_path}")

    return {
        "optimal_threshold": optimal_threshold,
        "test_results": test_results,
        "artifact_paths": {
            "class_names": config.class_names_path,
            "drift_baseline_stats": config.drift_baseline_stats_path,
            "pytorch_model": config.pytorch_model_path,
            "model_results": config.model_results_path,
        },
    }
