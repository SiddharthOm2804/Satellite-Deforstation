"""Train and calibrate EuroSAT classifier with raw [0, 255] inputs matching EfficientNetB0's native architecture."""

import io
import sys
from pathlib import Path
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from src.model import build_model
from src.data_loader import CLASS_NAMES

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR = Path("data/examples")
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[1/5] Loading EuroSAT parquet dataset...")
    parquet_path = hf_hub_download(
        "blanchon/EuroSAT_RGB",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    print(f"Total available images in parquet: {len(records)}")

    # Group by label
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    # Use 250 images per class for training (2500 total) and 100 images per class for testing (1000 total)
    TRAIN_PER_CLASS = 250
    TEST_PER_CLASS = 100

    print(f"\n[2/5] Preparing dataset ({TRAIN_PER_CLASS} train / {TEST_PER_CLASS} test per class)...")
    train_records = []
    test_records = []

    rng = random.Random(42)
    for lbl in range(10):
        items = list(by_label[lbl])
        rng.shuffle(items)

        train_items = items[:TRAIN_PER_CLASS]
        test_items = items[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]

        for item in train_items:
            train_records.append((lbl, item["image"]["bytes"]))
        for item in test_items:
            test_records.append((lbl, item["image"]["bytes"]))

        # Also save high-quality authentic example images for UI from the test partition
        cls_name = CLASS_NAMES[lbl]
        if cls_name in ["Forest", "AnnualCrop", "Residential", "Industrial", "River"]:
            img_b = test_items[0]["image"]["bytes"]
            ex_img = Image.open(io.BytesIO(img_b)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            ex_path = EXAMPLES_DIR / f"example_{cls_name.lower()}.jpg"
            ex_img.save(str(ex_path), quality=95)
            print(f"Saved authentic example: {ex_path}")

    # Thoroughly shuffle datasets
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    print(f"Total train images: {len(train_records)}, test images: {len(test_records)}")

    def load_and_preprocess(record_list):
        images = []
        labels = []
        for lbl, img_bytes in record_list:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            # Pass raw [0, 255] float32 pixels to let EfficientNetB0 perform internal Rescaling and Normalization
            arr = np.array(img, dtype=np.float32)
            images.append(arr)
            labels.append(lbl)
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    cache_file = RESULTS_DIR / "features_cache_raw255.npz"
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}...")
        cached = np.load(str(cache_file))
        X_train_feat = cached["X_train_feat"]
        y_train = cached["y_train"]
        X_test_feat = cached["X_test_feat"]
        y_test = cached["y_test"]
        full_model = build_model(input_shape=(224, 224, 3), num_classes=10)
    else:
        print("\n[3/5] Preprocessing and extracting features with EfficientNetB0 (raw [0, 255] RGB)...")
        X_train, y_train = load_and_preprocess(train_records)
        X_test, y_test = load_and_preprocess(test_records)

        full_model = build_model(input_shape=(224, 224, 3), num_classes=10)
        feat_extractor = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=full_model.get_layer("global_avg_pool").output,
        )

        print("Extracting train features...")
        X_train_feat = feat_extractor.predict(X_train, batch_size=64, verbose=1)
        print("Extracting test features...")
        X_test_feat = feat_extractor.predict(X_test, batch_size=64, verbose=1)
        np.savez_compressed(
            str(cache_file),
            X_train_feat=X_train_feat,
            y_train=y_train,
            X_test_feat=X_test_feat,
            y_test=y_test,
        )
        print(f"Cached raw features to {cache_file}")

    print(f"Feature shapes: Train {X_train_feat.shape}, Test {X_test_feat.shape}")

    print("\n[4/5] Training regularized multinomial classifier (L-BFGS)...")
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_train_feat, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_feat))
    test_acc = accuracy_score(y_test, clf.predict(X_test_feat))
    print(f"Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = clf.predict(X_test_feat)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nClassification Report on Unseen EuroSAT Test Set:")
    print(report)

    # Save classification report to results/
    report_file = RESULTS_DIR / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(f"EuroSAT Land Use Classification Report (EfficientNetB0 + Calibrated Head)\n")
        f.write(f"Overall Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved updated report to {report_file}")

    # Transfer trained weights to Keras model
    kernel_weights = clf.coef_.T.astype(np.float32)   # (1280, 10)
    bias_weights = clf.intercept_.astype(np.float32)  # (10,)

    pred_layer = full_model.get_layer("predictions")
    pred_layer.set_weights([kernel_weights, bias_weights])

    # Save calibrated model to results/best_model.h5
    model_save_path = RESULTS_DIR / "best_model.h5"
    full_model.save(str(model_save_path))
    print(f"\n[5/5] Successfully saved calibrated model to: {model_save_path}")

    # Test specifically on data/examples/
    print("\n" + "=" * 60)
    print("PREDICTIONS ON EXAMPLE IMAGES:")
    print("=" * 60)
    for name in ["example_forest.jpg", "example_annualcrop.jpg", "example_residential.jpg", "example_industrial.jpg", "example_river.jpg"]:
        p = EXAMPLES_DIR / name
        img = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        probs = full_model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_cls = CLASS_NAMES[top_idx]
        conf = probs[top_idx] * 100
        print(f"  {name:25s} -> {top_cls:20s} ({conf:5.1f}%)")


if __name__ == "__main__":
    main()
