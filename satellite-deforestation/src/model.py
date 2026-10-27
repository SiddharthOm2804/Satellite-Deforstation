"""EfficientNetB0 transfer learning architecture for EuroSAT land use classification.

Implements:
1. EfficientNetB0 with weights='imagenet', include_top=False, input_shape=(224, 224, 3).
2. Base model layers frozen initially.
3. Custom classification head:
   - GlobalAveragePooling2D()
   - Dropout(0.3)
   - Dense(10, activation='softmax')
4. Model compilation with Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'].
5. unfreeze_top_layers(model, n=20) for fine-tuning.
6. Returns compiled model.
"""

from typing import Tuple, Optional
import tensorflow as tf


def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Build and compile EfficientNetB0 transfer learning model.

    Args:
        input_shape: Input image dimensions (224, 224, 3).
        num_classes: Number of classification categories (10 for EuroSAT).
        learning_rate: Initial Adam learning rate (1e-4).
        dropout_rate: Dropout fraction (0.3).

    Returns:
        Compiled tf.keras.Model ready for Stage 1 training.
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")

    # 1. Load EfficientNetB0 backbone without top classification layers
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )

    # 2. Freeze all base layers initially
    base_model.trainable = False

    # 3. Add custom classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EuroSAT_EfficientNetB0")

    # 4. Compile model with Adam(1e-4) and categorical crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"[Model] Initialized EfficientNetB0 with {num_classes} classes. Base backbone frozen.")
    return model


def unfreeze_top_layers(
    model: tf.keras.Model,
    n: int = 20,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Unfreeze the top n layers of the EfficientNetB0 backbone for fine-tuning.

    Args:
        model: Compiled transfer learning model.
        n: Number of top layers to unfreeze (default 20).
        learning_rate: Fine-tuning learning rate (default 1e-5).

    Returns:
        Recompiled tf.keras.Model ready for Stage 2 fine-tuning.
    """
    # Locate base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name.lower():
            base_model = layer
            break

    target_layers = base_model.layers if base_model is not None else model.layers

    if base_model is not None:
        base_model.trainable = True

    # Freeze earlier layers
    for layer in target_layers[:-n]:
        layer.trainable = False

    # Unfreeze last n layers (preserving BatchNorm statistics in inference mode)
    unfrozen_count = 0
    for layer in target_layers[-n:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen_count += 1

    # Re-compile model with fine-tuning learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"[Model] Unfroze top {n} layers ({unfrozen_count} trainable weights, BatchNorm frozen). Recompiled with lr={learning_rate}.")
    return model


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Identify the target convolutional layer for Grad-CAM.

    For EfficientNetB0, this is typically 'top_conv'.
    """
    target_names = ["top_conv", "top_activation", "conv5_block3_out", "final_conv"]
    for name in target_names:
        try:
            model.get_layer(name)
            return name
        except (ValueError, KeyError):
            pass

    # Search inside base model if nested
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return sub_layer.name
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    return "top_conv"


# Aliases for convenience
create_model = build_model

if __name__ == "__main__":
    print("[Model] Testing build_model()...")
    m = build_model()
    m.summary(show_trainable=True)

    print("\n[Model] Testing unfreeze_top_layers(m, n=20)...")
    m = unfreeze_top_layers(m, n=20)
    print(f"[Model] Last conv layer detected: {get_last_conv_layer_name(m)}")
