"""Two-phase training script for EuroSAT land use classification using EfficientNetB0.

Implements:
1. Imports data_loader and model.
2. Loads train, val, test datasets.
3. Builds EfficientNetB0 model.
4. Train Phase 1: 10 epochs with base frozen.
   - Callbacks: EarlyStopping(patience=3), ModelCheckpoint(results/best_model.h5), ReduceLROnPlateau.
5. Train Phase 2: Unfreezes last 20 layers, fine-tunes 5 epochs with lr=1e-5.
6. Saves combined training history to results/history.json.
7. Plots accuracy and loss curves (train vs val) and saves to results/curves.png.
8. Prints final validation accuracy.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf

from src import data_loader
from src import model


def plot_curves(history: Dict[str, list], output_path: str = "results/curves.png", phase1_epochs: int = 10) -> None:
    """Plot training and validation accuracy/loss curves and save to disk."""
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Accuracy Plot
    axes[0].plot(epochs, history["accuracy"], "o-", label="Train Accuracy", color="#1f77b4", lw=2, markersize=4)
    axes[0].plot(epochs, history["val_accuracy"], "s-", label="Val Accuracy", color="#ff7f0e", lw=2, markersize=4)
    if phase1_epochs < len(epochs):
        axes[0].axvline(x=phase1_epochs + 0.5, color="gray", linestyle="--", label="Phase 2 Fine-Tuning")
    axes[0].set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=10)

    # 2. Loss Plot
    axes[1].plot(epochs, history["loss"], "o-", label="Train Loss", color="#1f77b4", lw=2, markersize=4)
    axes[1].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e", lw=2, markersize=4)
    if phase1_epochs < len(epochs):
        axes[1].axvline(x=phase1_epochs + 0.5, color="gray", linestyle="--", label="Phase 2 Fine-Tuning")
    axes[1].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Categorical Cross-Entropy", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=300)
    plt.close(fig)
    print(f"[Train] Training curves saved to: {out_file}")


def train(
    batch_size: int = 32,
    epochs_phase1: int = 10,
    epochs_phase2: int = 5,
    results_dir: str = "results",
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Execute complete two-phase training protocol."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # 2. Load train, val, test datasets
    print("[Train] Step 1/5: Loading EuroSAT datasets...")
    train_ds, val_ds, test_ds, class_names = data_loader.load_dataset(batch_size=batch_size)

    # 3. Build EfficientNetB0 model
    print("\n[Train] Step 2/5: Building EfficientNetB0 model with frozen base...")
    classifier = model.build_model(
        input_shape=(224, 224, 3),
        num_classes=len(class_names),
        learning_rate=1e-4,
        dropout_rate=0.3,
    )

    best_model_path = str(results_path / "best_model.h5")

    # 4. Train Phase 1: 10 epochs with base frozen
    print("\n" + "=" * 60)
    print(f"PHASE 1: Training Classification Head ({epochs_phase1} epochs, lr=1e-4)")
    print("=" * 60)

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history1 = classifier.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=callbacks_phase1,
    )

    actual_phase1_epochs = len(history1.history["loss"])

    # 5. Train Phase 2: Unfreeze last 20 layers, fine-tune 5 epochs with lr=1e-5
    history2 = None
    if epochs_phase2 > 0:
        print("\n" + "=" * 60)
        print(f"PHASE 2: Fine-Tuning Top 20 Layers ({epochs_phase2} epochs, lr=1e-5)")
        print("=" * 60)

        classifier = model.unfreeze_top_layers(classifier, n=20, learning_rate=1e-5)

        callbacks_phase2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history2 = classifier.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_phase2,
            callbacks=callbacks_phase2,
        )

    # 6. Combine and save training history to results/history.json
    combined_history: Dict[str, list] = {}
    metric_keys = ["loss", "accuracy", "val_loss", "val_accuracy"]

    for k in metric_keys:
        vals = [float(v) for v in history1.history.get(k, [])]
        if history2 is not None:
            vals += [float(v) for v in history2.history.get(k, [])]
        combined_history[k] = vals

    history_path = results_path / "history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"\n[Train] Training history saved to: {history_path}")

    # 7. Plot accuracy and loss curves and save to results/curves.png
    curves_path = str(results_path / "curves.png")
    plot_curves(combined_history, output_path=curves_path, phase1_epochs=actual_phase1_epochs)

    # 8. Print final validation accuracy
    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON VALIDATION SET")
    print("=" * 60)
    val_loss, val_acc = classifier.evaluate(val_ds, verbose=0)
    print(f"\n>>> Final Validation Accuracy: {val_acc * 100:.2f}% (Val Loss: {val_loss:.4f}) <<<\n")

    return classifier, combined_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 on EuroSAT")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--epochs1", type=int, default=10, help="Phase 1 frozen epochs (default: 10)")
    parser.add_argument("--epochs2", type=int, default=5, help="Phase 2 fine-tuning epochs (default: 5)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output artifacts")
    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        epochs_phase1=args.epochs1,
        epochs_phase2=args.epochs2,
        results_dir=args.results_dir,
    )
