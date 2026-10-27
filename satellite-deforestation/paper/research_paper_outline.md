# Explainable Deep Learning for Temporal Satellite Imagery Classification and Deforestation Monitoring: A Grad-CAM Augmented Approach

**Target Venues**: *IEEE Transactions on Geoscience and Remote Sensing (TGRS)* / *ISPRS Journal of Photogrammetry and Remote Sensing* / *Remote Sensing of Environment*

---

## Abstract
Deforestation accounts for substantial global carbon emissions, biodiversity loss, and climate degradation. While deep convolutional neural networks (CNNs) have shown remarkable accuracy in land-use and land-cover (LULC) classification from satellite imagery, their black-box nature hinders trustworthy deployment in legal enforcement, environmental policy-making, and conservation management. This paper presents an end-to-end explainable Earth observation framework for real-time deforestation monitoring. Leveraging the Sentinel-2-derived EuroSAT benchmark across 10 distinct land cover classes, we train an optimized transfer learning architecture (EfficientNet/ResNet). We integrate Gradient-weighted Class Activation Mapping (Grad-CAM) to generate spatial attribution heatmaps, demonstrating that the network's predictive decisions correlate with true canopy cover rather than sensor artifacts. Furthermore, we implement a bi-temporal change detection engine combining semantic transition analysis with pixel-level Green Leaf Index (GLI) differencing and morphological filtering to quantify canopy loss percentage. Empirical evaluations demonstrate competitive macro F1-scores, high recall on forest preservation classes, and verifiable spatial attention, bridging the gap between automated satellite vision and actionable environmental accountability.

---

## 1. Introduction
- **1.1 The Global Deforestation Crisis**:
  - Critical role of tropical and temperate forest biomes in terrestrial carbon sequestration.
  - Scale of anthropogenic clearing: agricultural encroachment, illegal logging, infrastructural expansion.
- **1.2 Remote Sensing Opportunities & Challenges**:
  - High revisit rates of ESA Sentinel-2 and NASA Landsat constellations.
  - Limitations of manual photogrammetry: labor intensity, temporal latency, subjective interpretation.
- **1.3 The Black-Box Dilemma in AI-Driven Earth Observation**:
  - High stakes of automated enforcement (e.g., sanctioning landholders or calculating carbon credits).
  - Risk of deep models learning spurious correlations (e.g., field borders, sensor noise, cloud shadows) rather than authentic canopy features.
- **1.4 Core Contributions**:
  1. *Transfer Learning Optimization*: High-performance classification across 10 EuroSAT land cover classes with stratified two-stage fine-tuning.
  2. *Spatial Explainability via Grad-CAM*: Class-specific visual attributions verifying that classification decisions stem from dense vegetation features.
  3. *Dual-Scale Deforestation Change Detection*: Synergistic bi-temporal pipeline detecting semantic transitions (Forest $\to$ Non-Forest) while computing physical canopy loss percentages.
  4. *Open-Source Interactive System*: Production-ready Gradio platform for interactive inspection and reproducible scientific exploration.

---

## 2. Related Work
- **2.1 Deep Learning for Land Use / Land Cover (LULC) Classification**:
  - Evolution from spectral index thresholding (NDVI, EVI) to deep CNNs and Vision Transformers (ViTs).
  - Benchmark datasets: UC Merced, AID, RESISC45, and EuroSAT.
- **2.2 Explainable Artificial Intelligence (XAI) in Remote Sensing**:
  - Post-hoc attribution methods: Saliency Maps, Integrated Gradients, SHAP, and Grad-CAM.
  - Unique challenges of overhead nadir imagery compared to standard egocentric computer vision (scale variance, lack of standard canonical orientation).
- **2.3 Satellite Change Detection Methodologies**:
  - Direct pixel differencing, Post-Classification Comparison (PCC), and Siamese feature extraction networks.
  - Trade-offs between computational overhead, false alarm rates from seasonal phenology, and actionable alert speed.

---

## 3. Dataset & Preprocessing Pipeline
- **3.1 EuroSAT Dataset Characteristics**:
  - Sentinel-2 satellite imagery covering 34 European countries.
  - 27,000 georeferenced patches ($64 \times 64$ pixels, 10m Ground Sample Distance).
  - 10 representative classes: *Forest* (preservation target), *AnnualCrop*, *HerbaceousVegetation*, *Highway*, *Industrial*, *Pasture*, *PermanentCrop*, *Residential*, *River*, *SeaLake*.
- **3.2 Data Augmentation & Invariance Modeling**:
  - Rotational symmetry: satellite nadir imagery has no canonical "upright" axis; random 4-fold flips and rotations prevent directional overfitting.
  - Photometric jitter: simulated variations in solar elevation angle and atmospheric attenuation.
- **3.3 Input Standardization**:
  - Normalization to $[0, 1]$ and bi-cubic upsampling to $128 \times 128$ for optimal convolutional receptive field coverage.

---

## 4. Methodology
```
                  ┌──────────────────────┐
                  │ Pre-event Image (T1) │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ CNN Classifier (T1)   │         │ Green Leaf Index (T1) │
└───────────┬───────────┘         └───────────┬───────────┘
            │ Class Probabilities             │
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ CNN Classifier (T2)   │         │ Green Leaf Index (T2) │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            ├───────────────┬─────────────────┘
            ▼               ▼
   [Semantic Shift]  [Δ Vegetation & Morphological Filter]
            │               │
            └───────┬───────┘
                    ▼
     [Deforestation Severity Index]
                    ▼
       [Grad-CAM Attribution Map]
```

- **4.1 Network Architecture & Transfer Learning**:
  - Backbone: Pre-trained EfficientNetB0 (Compound Scaling) and ResNet50V2.
  - Head: Global Average Pooling $\to$ Batch Normalization $\to$ Dense(256, ReLU) $\to$ Dropout(0.4) $\to$ Softmax(10).
- **4.2 Grad-CAM Mathematical Formulation**:
  - Gradient computation for target land cover class $c$:
    $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{ij}^k}$$
  - Localization map generation:
    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
  - Normalization and bilinear upsampling for superimposition onto original satellite patches.
- **4.3 Bi-Temporal Deforestation Change Detection Algorithm**:
  - Semantic Transition Condition: $\mathcal{C}(T_1) = \text{Forest} \land \mathcal{C}(T_2) \neq \text{Forest}$.
  - Visible Vegetation Density metric (Green Leaf Index):
    $$\text{GLI} = \frac{2G - R - B}{2G + R + B + \epsilon}$$
  - Morphological noise suppression: opening followed by closing with a $3 \times 3$ structuring element to filter registration jitter.
  - Deforestation Severity Index (DSI): Percentage of canopy lost over total patch area.

---

## 5. Experimental Design & Implementation
- **5.1 Training Protocol**:
  - Stage 1 (Warmup): 5 epochs, frozen backbone, Adam optimizer ($\eta = 10^{-3}$).
  - Stage 2 (Fine-Tuning): 10 epochs, top 35 layers unfrozen, Adam optimizer ($\eta = 5 \times 10^{-5}$) with Cosine Annealing / ReduceLROnPlateau.
  - Loss: Categorical Cross-Entropy with label smoothing ($\alpha = 0.05$).
- **5.2 Evaluation Metrics**:
  - Overall Accuracy ($OA$).
  - Macro and Weighted Precision, Recall, and F1-score.
  - Specific Forest Recall: minimizing False Negatives (missed deforestation).
  - Confusion Matrix analysis.

---

## 6. Results & Discussion
- **6.1 Quantitative Classification Performance**:
  - Benchmark comparison across backbone architectures.
  - Class-wise F1 breakdown highlighting separation between vegetative classes (*Forest* vs *Pasture* vs *HerbaceousVegetation*).
- **6.2 Qualitative Grad-CAM Validation**:
  - Heatmaps confirm that activation peaks align with tree cluster texture rather than background soil.
  - Counterfactual inspection: demonstrating what features lead to non-forest classifications.
- **6.3 Temporal Change Detection Case Studies**:
  - Simulated and empirical before/after deforestation pairs.
  - Verification that the combination of semantic shift + GLI filtering eliminates false alarms caused by lighting differences.

---

## 7. Limitations & Operational Considerations
- **7.1 Spectral Resolution**:
  - RGB limitations vs full 13-band Sentinel-2 multispectral imagery (B8 NIR, B11/B12 SWIR for true NDVI/NDWI).
- **7.2 Atmospheric Obstruction**:
  - Cloud cover, haze, and shadow artifacts requiring atmospheric correction (Sen2Cor) or cloud-masking preprocessing.
- **7.3 Spatial Granularity**:
  - 10m GSD cannot resolve individual tree logging, suitable primarily for contiguous clear-cutting (> 0.05 ha).

---

## 8. Conclusion & Future Work
- Summary of findings: Explainability and temporal tracking significantly improve confidence in autonomous Earth observation systems.
- Future roadmap: Integration of temporal self-attention (Temporal Transformers) and expansion to multispectral satellite constellations.

---

## References
1. Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification*. IEEE JSTARS.
2. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. ICCV.
3. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. ICML.
4. Hansen, M. C., et al. (2013). *High-resolution global maps of 21st-century forest cover change*. Science.
5. Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully convolutional Siamese networks for change detection*. ICIP.
