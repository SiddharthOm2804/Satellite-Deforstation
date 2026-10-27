"""Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNetB0 EuroSAT classifier.

Implements:
1. make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv').
2. Uses tf.GradientTape to compute gradients of predicted class w.r.t. last conv layer output.
3. Applies ReLU, normalizes to [0, 1], resizes to 224x224 using cv2.
4. Overlays heatmap on original image using cv2.applyColorMap with JET colormap.
5. Saves individual outputs to results/gradcam_outputs/ folder.
6. Function generate_gradcam_grid() generating Grad-CAM for 10 test images (one per class) in a grid.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import CLASS_NAMES, unnormalize_imagenet, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_CONV_LAYER: str = "top_conv"


def make_gradcam_heatmap(
    img_array: Union[np.ndarray, tf.Tensor],
    model: tf.keras.Model,
    last_conv_layer_name: str = DEFAULT_CONV_LAYER,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for an input image.

    Args:
        img_array: Preprocessed input tensor of shape (1, 224, 224, 3).
        model: Trained tf.keras.Model.
        last_conv_layer_name: Name of target conv layer ('top_conv' for EfficientNetB0).
        pred_index: Target class index. If None, top-1 predicted class is used.

    Returns:
        2D numpy array heatmap of shape (224, 224) with values in range [0, 1].
    """
    if isinstance(img_array, np.ndarray):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    else:
        img_tensor = img_array

    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)

    # 1. Locate last conv layer
    target_layer = model.get_layer(last_conv_layer_name)

    # 2. Build gradient model mapping inputs to [conv_activations, predictions]
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    # 3. Compute gradients using tf.GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of class score w.r.t. feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients: weights alpha_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels of the feature map
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 4. Apply ReLU, normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    # 5. Resize to (224, 224) using cv2
    heatmap_resized = cv2.resize(heatmap_np, (224, 224))
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    return heatmap_resized


def overlay_gradcam(
    img_raw: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Superimpose Grad-CAM heatmap onto original RGB image using JET colormap.

    Args:
        img_raw: Original RGB image array (float [0, 1] or uint8 [0, 255]) of shape (224, 224, 3).
        heatmap: 2D float heatmap in range [0, 1] of shape (224, 224).
        alpha: Blend weight for heatmap overlay (default: 0.45).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        Tuple of (colored_heatmap_rgb, blended_overlay_rgb) as uint8 arrays.
    """
    # Scale heatmap to [0, 255] uint8
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply JET colormap (cv2 outputs BGR)
    jet_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)

    # Convert img_raw to uint8 RGB
    if img_raw.dtype != np.uint8:
        if img_raw.max() <= 1.0:
            img_uint8 = np.uint8(255 * np.clip(img_raw, 0.0, 1.0))
        else:
            img_uint8 = np.uint8(np.clip(img_raw, 0, 255))
    else:
        img_uint8 = img_raw

    # Resize img_uint8 if not 224x224
    if img_uint8.shape[:2] != (224, 224):
        img_uint8 = cv2.resize(img_uint8, (224, 224))

    # Superimpose: alpha * heatmap + (1 - alpha) * original
    overlay = cv2.addWeighted(jet_rgb, alpha, img_uint8, 1.0 - alpha, 0)

    return jet_rgb, overlay


def save_gradcam(
    overlay: np.ndarray,
    output_path: str,
) -> str:
    """Save Grad-CAM overlay image to disk."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(str(out_file), quality=95)
    return str(out_file)


def generate_gradcam_for_image(
    image_input: Union[str, np.ndarray, Image.Image],
    model: tf.keras.Model,
    last_conv_layer_name: str = DEFAULT_CONV_LAYER,
    target_class_idx: Optional[int] = None,
    output_dir: str = "results/gradcam_outputs",
    filename_prefix: str = "gradcam_sample",
) -> Dict[str, Any]:
    """Generate and save Grad-CAM explanation for a single image."""
    # Preprocess image
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    pil_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_raw = np.array(pil_resized, dtype=np.float32) / 255.0

    # ImageNet normalization for model input
    img_norm = (img_raw - IMAGENET_MEAN.numpy()) / IMAGENET_STD.numpy()
    img_tensor = np.expand_dims(img_norm, axis=0)

    # Forward pass predictions
    preds = model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = float(preds[pred_idx])

    explain_idx = target_class_idx if target_class_idx is not None else pred_idx

    # Compute heatmap & overlay
    heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=explain_idx)
    jet_rgb, overlay = overlay_gradcam(img_raw, heatmap, alpha=0.45)

    # Save overlay to results/gradcam_outputs/
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_prefix}_{pred_class.lower()}.png"
    save_gradcam(overlay, str(out_path))

    return {
        "predicted_class": pred_class,
        "predicted_idx": pred_idx,
        "confidence": pred_conf,
        "heatmap": heatmap,
        "jet_rgb": jet_rgb,
        "overlay": overlay,
        "original": np.uint8(255 * img_raw),
        "saved_path": str(out_path),
    }


def generate_gradcam_grid(
    model: tf.keras.Model,
    dataset: Optional[tf.data.Dataset] = None,
    last_conv_layer_name: str = DEFAULT_CONV_LAYER,
    output_dir: str = "results/gradcam_outputs",
    grid_filename: str = "gradcam_10_classes_grid.png",
) -> plt.Figure:
    """Generate Grad-CAM for 10 test images (one per class) and save as a high-resolution grid.

    Extracts 1 test image per EuroSAT class, computes Grad-CAM heatmaps, overlays with JET colormap,
    saves individual outputs to results/gradcam_outputs/, and compiles a 10-row x 3-column figure:
    [Original Satellite Patch | Grad-CAM JET Heatmap | Superimposed Overlay].
    """
    out_path_dir = Path(output_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GradCAM] Collecting 1 test sample for each of the 10 classes...")
    collected_images: Dict[int, np.ndarray] = {}

    # Extract 1 sample per class from dataset
    if dataset is not None:
        unbatched = dataset.unbatch()
        for img, lbl in unbatched:
            lbl_idx = int(lbl.numpy()) if tf.rank(lbl) == 0 else int(tf.argmax(lbl).numpy())
            if lbl_idx not in collected_images:
                # Unnormalize to raw [0, 1]
                img_unnorm = unnormalize_imagenet(img).numpy()
                collected_images[lbl_idx] = img_unnorm
            if len(collected_images) == len(CLASS_NAMES):
                break
    else:
        # Load from TFDS directly
        import tensorflow_datasets as tfds
        test_raw = tfds.load("eurosat/rgb", split="train[85%:]", as_supervised=True, shuffle_files=True)
        for img, lbl in test_raw:
            lbl_idx = int(lbl.numpy())
            if lbl_idx not in collected_images:
                resized = tf.image.resize(img, (224, 224))
                img_float = (resized.numpy() / 255.0).clip(0.0, 1.0)
                collected_images[lbl_idx] = img_float
            if len(collected_images) == len(CLASS_NAMES):
                break

    # Build 10 rows x 3 columns plot
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(10, 26))
    fig.suptitle(
        "Explainable AI (Grad-CAM) - Spatial Attention on EuroSAT (10 Classes)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    column_titles = ["Original Satellite Patch", "Grad-CAM Heatmap", "JET Colormap Overlay"]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row_idx, class_name in enumerate(CLASS_NAMES):
        img_raw = collected_images.get(row_idx)
        if img_raw is None:
            continue

        # Prepare normalized tensor
        img_norm = (img_raw - IMAGENET_MEAN.numpy()) / IMAGENET_STD.numpy()
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_norm, axis=0), dtype=tf.float32)

        # Predict
        probs = model.predict(img_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        # Compute Grad-CAM
        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=row_idx)
        jet_rgb, overlay = overlay_gradcam(img_raw, heatmap, alpha=0.45)

        # Save individual overlay to results/gradcam_outputs/
        single_path = out_path_dir / f"gradcam_class_{row_idx:02d}_{class_name.lower()}.png"
        save_gradcam(overlay, str(single_path))

        # Plot Row
        # Col 0: Original
        axes[row_idx, 0].imshow(img_raw)
        axes[row_idx, 0].set_ylabel(
            f"{class_name}\n(Pred: {pred_class}, {conf:.1f}%)",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            labelpad=65,
            ha="right",
            va="center",
        )
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Col 1: Heatmap
        axes[row_idx, 1].imshow(jet_rgb)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Col 2: Overlay
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

    plt.tight_layout()

    grid_path = out_path_dir / grid_filename
    fig.savefig(str(grid_path), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[GradCAM] 10-Class Grad-CAM Grid saved to: {grid_path}")
    print(f"[GradCAM] Individual overlays saved in: {out_path_dir}")
    return fig


if __name__ == "__main__":
    from src.model import build_model
    print("[GradCAM] Testing Grad-CAM pipeline...")
    m = build_model()
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    hm = make_gradcam_heatmap(dummy, m, last_conv_layer_name="top_conv")
    print(f"[GradCAM] Heatmap shape: {hm.shape}, range: [{hm.min():.3f}, {hm.max():.3f}]")
    jet, ovr = overlay_gradcam(dummy[0], hm)
    print(f"[GradCAM] Overlay shape: {ovr.shape}, type: {ovr.dtype}")
    print("[GradCAM] Test passed successfully!")
