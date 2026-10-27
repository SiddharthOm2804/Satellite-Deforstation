"""Explainable Satellite Image Classification System for Deforestation Monitoring.

Modules:
    data_loader: Dataset downloading, splitting, and tf.data preprocessing.
    model: CNN architecture definitions and transfer learning backbones.
    train: Training loops, callbacks, and metric tracking.
    evaluate: Model validation, classification metrics, and confusion matrix.
    gradcam: Gradient-weighted Class Activation Mapping explainability engine.
    change_detection: Bi-temporal satellite change analysis for deforestation.
    app: Interactive Gradio web application.
"""

__version__ = "1.0.0"
