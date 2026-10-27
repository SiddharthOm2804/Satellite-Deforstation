"""EuroSAT dataset loader, preprocessing pipeline, and sample visualizer using TensorFlow Datasets (TFDS).

Implements:
1. tfds.load('eurosat/rgb', split='train', as_supervised=True).
2. EuroSAT 10 land cover class names.
3. 70% Train, 15% Validation, 15% Test stratified sub-splits.
4. Resizing to 224x224.
5. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
6. Data augmentation: rotation_range=20, horizontal_flip=True, zoom_range=0.1, brightness_range=[0.8, 1.2].
7. Batched tf.data.Dataset pipelines (batch_size=32).
8. show_samples() displaying a 10x4 grid of 4 samples per class.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# 10 Official EuroSAT Classes
CLASS_NAMES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# ImageNet normalization statistics
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)


def get_class_names() -> List[str]:
    """Return the 10 EuroSAT class names."""
    return list(CLASS_NAMES)


def build_augmentation_layer() -> tf.keras.Sequential:
    """Build data augmentation pipeline.

    Applies:
    - rotation_range=20 degrees (20 / 360 factor)
    - horizontal_flip=True
    - zoom_range=0.1 (height & width factor (-0.1, 0.1))
    - brightness_range=[0.8, 1.2] (RandomBrightness factor 0.2)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=20.0 / 360.0, fill_mode="reflect"),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect"),
            tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),
        ],
        name="data_augmentation",
    )


def normalize_imagenet(image: tf.Tensor) -> tf.Tensor:
    """Normalize image tensor using ImageNet mean and std.

    Args:
        image: Float32 image tensor in range [0.0, 1.0].

    Returns:
        Standardized image tensor.
    """
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def unnormalize_imagenet(image: tf.Tensor) -> tf.Tensor:
    """Reverse ImageNet standardization for visualization."""
    unnorm = (image * IMAGENET_STD) + IMAGENET_MEAN
    return tf.clip_by_value(unnorm, 0.0, 1.0)


def preprocess_image(
    image: tf.Tensor,
    label: tf.Tensor,
    img_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    augment: bool = False,
    augmentation_layer: Optional[tf.keras.layers.Layer] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Resize, optionally augment, and standardize image."""
    # Resize to 224x224
    image = tf.image.resize(image, img_size)

    # Scale raw uint8 [0, 255] to float32 [0.0, 1.0]
    image = tf.cast(image, tf.float32) / 255.0

    # Apply data augmentation if training
    if augment and augmentation_layer is not None:
        # Augmentation expects batch or 3D tensor in [0, 1]
        image = augmentation_layer(image, training=True)
        image = tf.clip_by_value(image, 0.0, 1.0)

    # Standardize using ImageNet statistics
    image = normalize_imagenet(image)

    # Convert integer class label to one-hot encoding for categorical_crossentropy
    if tf.rank(label) == 0 or (tf.is_tensor(label) and label.shape.ndims and label.shape[-1] != len(CLASS_NAMES)):
        label = tf.one_hot(tf.cast(label, tf.int32), depth=len(CLASS_NAMES))

    return image, label


def load_dataset(
    batch_size: int = 32,
    img_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    data_dir: Optional[str] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load EuroSAT using TFDS, split into 70/15/15, preprocess, and batch.

    1. Load EuroSAT dataset using tfds.load('eurosat/rgb', split='train', as_supervised=True).
    2. Split into train (70%), validation (15%), test (15%).
    3. Resize all images to 224x224.
    4. Normalize using ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    5. Apply data augmentation to train set.
    6. Return tf.data.Dataset objects for train, val, test with batch_size=32.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names).
    """
    print("[DataLoader] Loading EuroSAT via tfds.load('eurosat/rgb', split='train', as_supervised=True)...")

    # Load 70% Train, 15% Validation, 15% Test sub-splits directly via TFDS
    split_spec = ["train[:70%]", "train[70%:85%]", "train[85%:]"]

    splits, info = tfds.load(
        "eurosat/rgb",
        split=split_spec,
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
    )

    train_raw, val_raw, test_raw = splits[0], splits[1], splits[2]
    class_names = info.features["label"].names

    print(f"[DataLoader] Successfully loaded EuroSAT. Classes: {class_names}")

    aug_layer = build_augmentation_layer()

    # Preprocessing pipelines
    train_ds = (
        train_raw.map(
            lambda img, lbl: preprocess_image(img, lbl, img_size=img_size, augment=True, augmentation_layer=aug_layer),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_raw.map(
            lambda img, lbl: preprocess_image(img, lbl, img_size=img_size, augment=False),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_raw.map(
            lambda img, lbl: preprocess_image(img, lbl, img_size=img_size, augment=False),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names


def show_samples(
    dataset: Optional[tf.data.Dataset] = None,
    class_names: Optional[List[str]] = None,
    samples_per_class: int = 4,
    save_path: Optional[str] = "results/plots/eurosat_sample_grid.png",
) -> plt.Figure:
    """Display 4 sample images per class in a clean 10x4 grid.

    Args:
        dataset: tf.data.Dataset yielding (image, label).
                 If None, loads raw unbatched EuroSAT samples.
        class_names: List of class names (defaults to EuroSAT classes).
        samples_per_class: Number of samples per class (default 4).
        save_path: Optional file path to save figure.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = len(class_names)
    collected_samples: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_classes)}
    needed_total = num_classes * samples_per_class

    print(f"[DataLoader] Collecting {samples_per_class} sample images per class across {num_classes} classes...")

    # If dataset not provided, load raw un-normalized images for clearest visual display
    if dataset is None:
        raw_ds = tfds.load("eurosat/rgb", split="train", as_supervised=True, shuffle_files=True)
        for img, lbl in raw_ds:
            lbl_idx = int(lbl.numpy())
            if len(collected_samples[lbl_idx]) < samples_per_class:
                # Resize to 224x224 and scale to float [0, 1]
                img_resized = tf.image.resize(img, TARGET_IMAGE_SIZE)
                img_np = (img_resized.numpy() / 255.0).clip(0.0, 1.0)
                collected_samples[lbl_idx].append(img_np)

            # Check if all classes have enough samples
            if sum(len(v) for v in collected_samples.values()) >= needed_total:
                break
    else:
        # Extract from provided tf.data.Dataset (unnormalize ImageNet stats)
        unbatched = dataset.unbatch()
        for img, lbl in unbatched:
            lbl_idx = int(lbl.numpy()) if tf.rank(lbl) == 0 else int(tf.argmax(lbl).numpy())
            if len(collected_samples[lbl_idx]) < samples_per_class:
                img_unnorm = unnormalize_imagenet(img).numpy()
                collected_samples[lbl_idx].append(img_unnorm)

            if sum(len(v) for v in collected_samples.values()) >= needed_total:
                break

    # Build 10 rows x 4 columns grid
    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=samples_per_class,
        figsize=(samples_per_class * 2.8, num_classes * 2.5),
    )

    fig.suptitle(
        "EuroSAT Land Cover Dataset - 4 Sample Images Per Class (224x224)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, class_name in enumerate(class_names):
        samples = collected_samples[row_idx]
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                ax.imshow(samples[col_idx])
            else:
                ax.axis("off")
                continue

            ax.set_xticks([])
            ax.set_yticks([])

            # Label class name on first column
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=12, fontweight="bold", rotation=0, labelpad=70, ha="right", va="center")

            if row_idx == 0:
                ax.set_title(f"Sample {col_idx + 1}", fontsize=11, pad=6)

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_file), dpi=200, bbox_inches="tight")
        print(f"[DataLoader] Sample grid saved to: {out_file}")

    return fig


if __name__ == "__main__":
    print("[DataLoader] Initializing EuroSAT pipeline test...")
    train_ds, val_ds, test_ds, classes = load_dataset(batch_size=32)
    print(f"[DataLoader] Train dataset: {train_ds}")
    print(f"[DataLoader] Generating sample grid...")
    fig = show_samples(save_path="results/plots/eurosat_samples_grid.png")
    plt.close(fig)
    print("[DataLoader] Done!")
