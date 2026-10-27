"""Train classification head on real EuroSAT Sentinel-2 images downloaded from Hugging Face parquet."""

import io
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import tensorflow as tf

from src.model import build_model
from src.data_loader import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR = Path("data/examples")
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[RealEuroSAT] Loading real EuroSAT dataset from Hugging Face cache...")
    parquet_path = hf_hub_download(
        "blanchon/EuroSAT_RGB",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    print(f"[RealEuroSAT] Total records in parquet: {len(records)}")

    # Group by label
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    print(f"[RealEuroSAT] Available classes: {len(by_label)}")
    for lbl, items in sorted(by_label.items()):
        print(f"  Class {lbl:02d} ({CLASS_NAMES[lbl]}): {len(items)} images")

    # Select 120 samples per class for training/validation (1,200 total real images)
    SAMPLES_PER_CLASS = 120
    X_train_raw = []
    y_train = []

    # Save authentic examples for Gradio UI
    for lbl in range(len(CLASS_NAMES)):
        items = by_label[lbl][:SAMPLES_PER_CLASS]
        cls_name = CLASS_NAMES[lbl]

        for idx, item in enumerate(items):
            img_bytes = item["image"]["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_224 = pil_img.resize((224, 224), Image.Resampling.BILINEAR)

            # Save first sample for each example class in Gradio
            if idx == 0 and cls_name in ["Forest", "AnnualCrop", "Residential", "Industrial", "River"]:
                ex_path = EXAMPLES_DIR / f"example_{cls_name.lower()}.jpg"
                pil_224.save(str(ex_path), quality=95)
                print(f"[RealEuroSAT] Saved real example: {ex_path}")

            arr = np.array(pil_224, dtype=np.float32) / 255.0
            X_train_raw.append(arr)
            y_train.append(lbl)

    X_train_raw = np.array(X_train_raw, dtype=np.float32)
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASS_NAMES))

    # Standardize using ImageNet statistics
    X_train_norm = (X_train_raw - IMAGENET_MEAN.numpy()) / IMAGENET_STD.numpy()

    print(f"\n[RealEuroSAT] Prepared {len(X_train_norm)} real EuroSAT images. Building EfficientNetB0 backbone...")
    full_model = build_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES), learning_rate=1e-3)

    # Pre-extract 1280-D features from EfficientNetB0
    print("[RealEuroSAT] Extracting features from pre-trained backbone...")
    feature_extractor = tf.keras.Model(
        inputs=full_model.inputs,
        outputs=full_model.get_layer("global_avg_pool").output,
    )
    features = feature_extractor.predict(X_train_norm, batch_size=32, verbose=1)

    # Train Dense classification head
    print("[RealEuroSAT] Training classification head on REAL satellite features...")
    head_inputs = tf.keras.layers.Input(shape=(1280,))
    h = tf.keras.layers.Dropout(0.3)(head_inputs)
    h_out = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="predictions")(h)
    head_model = tf.keras.Model(inputs=head_inputs, outputs=h_out)

    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    head_model.fit(
        features,
        y_train_onehot,
        epochs=35,
        batch_size=32,
        validation_split=0.15,
        verbose=1,
    )

    # Transfer trained weights to full model
    dense_weights = head_model.get_layer("predictions").get_weights()
    full_model.get_layer("predictions").set_weights(dense_weights)

    # Save to results/best_model.h5
    best_model_h5 = RESULTS_DIR / "best_model.h5"
    full_model.save(str(best_model_h5))
    print(f"\n[RealEuroSAT] Successfully trained and saved real model to: {best_model_h5}")

    # Evaluate predictions on held-out samples for all classes
    print("\n" + "=" * 60)
    print("VERIFYING ACCURACY ON REAL HOLDOUT SAMPLES:")
    print("=" * 60)
    test_correct = 0
    test_total = 0

    for lbl in range(len(CLASS_NAMES)):
        cls_name = CLASS_NAMES[lbl]
        # Use samples that were not in the training 120
        holdout_items = by_label[lbl][SAMPLES_PER_CLASS:SAMPLES_PER_CLASS + 20]
        cls_hits = 0

        for item in holdout_items:
            img_bytes = item["image"]["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            arr_norm = (arr - IMAGENET_MEAN.numpy()) / IMAGENET_STD.numpy()
            pred_probs = full_model.predict(np.expand_dims(arr_norm, axis=0), verbose=0)[0]
            pred_lbl = int(np.argmax(pred_probs))
            if pred_lbl == lbl:
                cls_hits += 1

        test_correct += cls_hits
        test_total += len(holdout_items)
        print(f"  * Class {cls_name:22s}: {cls_hits}/{len(holdout_items)} correct ({cls_hits/len(holdout_items)*100:.1f}%)")

    print("-" * 60)
    print(f"Total Holdout Test Accuracy on Real EuroSAT: {test_correct}/{test_total} ({test_correct/test_total*100:.2f}%)")
    print("-" * 60)


if __name__ == "__main__":
    main()
