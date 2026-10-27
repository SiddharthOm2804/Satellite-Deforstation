"""Bi-temporal satellite change detection module for deforestation monitoring.

Implements:
1. detect_change(img_2020, img_2024, model) taking two images as input.
2. Uses trained model to predict class and confidence for both images.
3. Compares predictions:
   - If before == "Forest" and after in ["Residential", "Industrial", "AnnualCrop", ...] -> "DEFORESTATION DETECTED"
   - If before == "Forest" and after == "Forest" -> "No Change"
   - Else -> "Other Change"
4. Generates side-by-side visualization: [2020 image | prediction] vs [2024 image | prediction] with status banner.
5. Saves visualization to results/change_detection/ folder.
6. process_change_detection_batch() processes a list of image pairs in batch.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

DEFORESTATION_TARGETS = [
    "Residential",
    "Industrial",
    "AnnualCrop",
    "Highway",
    "Pasture",
    "PermanentCrop",
]


def load_and_preprocess_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and return both unnormalized RGB [0, 1] and ImageNet normalized tensor.

    Returns:
        Tuple of (img_raw_float [0, 1], img_norm_tensor (1, 224, 224, 3)).
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_raw_255 = np.array(pil_resized, dtype=np.float32)
    img_display = img_raw_255 / 255.0  # [0, 1] for matplotlib display

    # EfficientNetB0 has internal rescaling and normalization
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_raw_255, axis=0), dtype=tf.float32)

    return img_display, img_tensor


def detect_change(
    img_2020: Union[str, Path, np.ndarray, Image.Image],
    img_2024: Union[str, Path, np.ndarray, Image.Image],
    model: tf.keras.Model,
    output_dir: str = "results/change_detection",
    save_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare bi-temporal satellite images (2020 vs 2024) to detect deforestation.

    Args:
        img_2020: Baseline satellite image (2020 / Before).
        img_2024: Post-event satellite image (2024 / After).
        model: Trained land use classifier.
        output_dir: Folder to save visualization.
        save_filename: Optional custom output filename.

    Returns:
        Dict with predicted classes, confidences, change status, and saved image path.
    """
    # 1. Preprocess both images
    img_raw_2020, tensor_2020 = load_and_preprocess_image(img_2020)
    img_raw_2024, tensor_2024 = load_and_preprocess_image(img_2024)

    # 2. Predict classes with trained model
    batch = tf.concat([tensor_2020, tensor_2024], axis=0)
    predictions = model.predict(batch, verbose=0)

    probs_2020, probs_2024 = predictions[0], predictions[1]
    idx_2020 = int(np.argmax(probs_2020))
    idx_2024 = int(np.argmax(probs_2024))

    pred_2020 = CLASS_NAMES[idx_2020]
    pred_2024 = CLASS_NAMES[idx_2024]
    conf_2020 = float(probs_2020[idx_2020])
    conf_2024 = float(probs_2024[idx_2024])

    # 3. Compare predictions according to deforestation rules
    # - If before = "Forest" and after = "Residential/Industrial/AnnualCrop" (or converted) -> DEFORESTATION DETECTED
    # - If before = "Forest" and after = "Forest" -> No Change
    # - Else -> Other Change
    if pred_2020 == "Forest" and pred_2024 in DEFORESTATION_TARGETS:
        change_status = "DEFORESTATION DETECTED"
        banner_bg = "#dc2626"   # Crimson Red
        banner_text_color = "white"
        status_icon = "🚨"
    elif pred_2020 == "Forest" and pred_2024 == "Forest":
        change_status = "No Change"
        banner_bg = "#16a34a"   # Forest Green
        banner_text_color = "white"
        status_icon = "🌲"
    else:
        change_status = "Other Change"
        banner_bg = "#2563eb"   # Royal Blue
        banner_text_color = "white"
        status_icon = "ℹ️"

    # 4. Generate side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: 2020 Image
    axes[0].imshow(img_raw_2020)
    axes[0].set_title(
        f"2020 Baseline (Before)\nPrediction: {pred_2020} ({conf_2020 * 100:.1f}%)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Subplot 2: 2024 Image
    axes[1].imshow(img_raw_2024)
    axes[1].set_title(
        f"2024 Observation (After)\nPrediction: {pred_2024} ({conf_2024 * 100:.1f}%)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Prominent Banner across the top
    banner_label = f"{status_icon} STATUS: {change_status} ({pred_2020} -> {pred_2024})"
    fig.text(
        0.5,
        0.96,
        banner_label,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=banner_text_color,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=banner_bg,
            edgecolor="none",
            alpha=0.95,
        ),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # 5. Save visualization to results/change_detection/
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_filename is None:
        save_filename = f"change_{pred_2020.lower()}_to_{pred_2024.lower()}.png"

    save_path = out_dir / save_filename
    fig.savefig(str(save_path), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[ChangeDetection] Visualization saved to: {save_path}")

    return {
        "pred_2020": pred_2020,
        "conf_2020": conf_2020,
        "pred_2024": pred_2024,
        "conf_2024": conf_2024,
        "change_status": change_status,
        "saved_path": str(save_path),
    }


def process_change_detection_batch(
    image_pairs: List[Tuple[Union[str, Path, np.ndarray, Image.Image], Union[str, Path, np.ndarray, Image.Image]]],
    model: tf.keras.Model,
    output_dir: str = "results/change_detection",
    summary_filename: str = "batch_change_summary.csv",
) -> pd.DataFrame:
    """Process a list of bi-temporal satellite image pairs in batch.

    Args:
        image_pairs: List of (img_2020, img_2024) image inputs.
        model: Trained classifier.
        output_dir: Output folder for visualizations and summary table.
        summary_filename: CSV filename for batch summary.

    Returns:
        pandas.DataFrame summarizing all evaluated pairs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    print(f"\n[ChangeDetection] Processing {len(image_pairs)} bi-temporal image pairs in batch...")

    for idx, (img_2020, img_2024) in enumerate(image_pairs, 1):
        filename = f"pair_{idx:03d}.png"
        res = detect_change(
            img_2020=img_2020,
            img_2024=img_2024,
            model=model,
            output_dir=str(out_dir),
            save_filename=filename,
        )

        results.append(
            {
                "Pair ID": idx,
                "2020 (Before)": res["pred_2020"],
                "2020 Conf (%)": round(res["conf_2020"] * 100, 2),
                "2024 (After)": res["pred_2024"],
                "2024 Conf (%)": round(res["conf_2024"] * 100, 2),
                "Change Status": res["change_status"],
                "Visualization File": filename,
            }
        )

    df_summary = pd.DataFrame(results)
    csv_path = out_dir / summary_filename
    df_summary.to_csv(str(csv_path), index=False)
    print(f"[ChangeDetection] Batch processing complete. Summary saved to: {csv_path}")

    # Deforestation Summary Metrics
    total = len(df_summary)
    deforested = len(df_summary[df_summary["Change Status"] == "DEFORESTATION DETECTED"])
    no_change = len(df_summary[df_summary["Change Status"] == "No Change"])
    other = len(df_summary[df_summary["Change Status"] == "Other Change"])

    print("-" * 65)
    print(f"[ChangeDetection] Batch Evaluation Summary (Total Pairs: {total}):")
    print(f"   * DEFORESTATION DETECTED: {deforested} ({(deforested/total)*100:.1f}%)")
    print(f"   * No Change (Canopy Intact): {no_change} ({(no_change/total)*100:.1f}%)")
    print(f"   * Other Change: {other} ({(other/total)*100:.1f}%)")
    print("-" * 65)

    return df_summary


if __name__ == "__main__":
    from src.model import build_model
    print("[ChangeDetection] Testing change detection module...")
    m = build_model()
    d1 = np.random.rand(224, 224, 3).astype(np.float32)
    d2 = np.random.rand(224, 224, 3).astype(np.float32)
    res = detect_change(d1, d2, m, save_filename="test_change.png")
    print(f"[ChangeDetection] Test result: {res['change_status']} (Saved: {res['saved_path']})")
