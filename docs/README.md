# Technical & Academic Documentation
## Crop Disease Detection using Convolutional Neural Network (CNN)

This document provides in-depth theoretical explanations, architectural formulas, and viva preparation notes for students presenting this capstone project.

---

## 1. Deep Learning & Computer Vision Foundations

### 1.1 Why Traditional Machine Learning (Handcrafted Features) Falls Short
Traditional approaches (e.g. SVM, Random Forest, or k-NN on color histograms or GLCM texture features) fail in real agricultural environments due to:
1. **Intra-class variability**: Leaves under different sunlight, angles, or growth stages exhibit varying hues.
2. **Inter-class visual similarity**: Early blight and bacterial spot both produce dark foliar necrotic spots that appear virtually identical in global color histograms.
3. **Loss of spatial hierarchy**: Traditional feature extractors discard spatial pixel relations, whereas CNNs learn multi-scale spatial representations automatically from raw data.

### 1.2 Mathematical Formulation of Convolution
For a 2D image $I$ and a kernel $K$ of dimensions $(2m+1) \times (2n+1)$:
$$(I * K)(i, j) = \sum_{u=-m}^{m} \sum_{v=-n}^{n} I(i - u, j - v) \cdot K(u, v)$$

In deep CNNs with $C_{in}$ input channels and $C_{out}$ output feature maps:
$$Y_k = \sum_{c=1}^{C_{in}} X_c * W_{k, c} + b_k$$

### 1.3 Inverted Residuals & Linear Bottlenecks (MobileNetV2)
Standard convolutions apply spatial filtering and channel transformation simultaneously, requiring $D_K \times D_K \times M \times N \times D_F \times D_F$ operations.
MobileNetV2 decomposes this into:
1. **Depthwise Convolution**: Applies a single spatial filter per input channel:
   $$\text{FLOPs}_{depthwise} = D_K \times D_K \times M \times D_F \times D_F$$
2. **Pointwise Convolution**: Applies $1 \times 1$ convolutions across channels:
   $$\text{FLOPs}_{pointwise} = 1 \times 1 \times M \times N \times D_F \times D_F$$
**Total Reduction Ratio**:
$$\frac{D_K \cdot D_K \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$
For a $3 \times 3$ kernel ($D_K=3$), this yields an **8- to 9-fold computational saving** with comparable classification accuracy.

---

## 2. Loss Function & Optimization

### 2.1 Sparse Categorical Crossentropy
Given ground truth class label $y \in \{0, 1, \dots, C-1\}$ and predicted probability distribution $\hat{y}$:
$$\mathcal{L}_{CE} = - \log(\hat{y}_y) = - \log \left( \frac{\exp(z_y)}{\sum_{j=1}^{C} \exp(z_j)} \right)$$

### 2.2 Adam Optimizer (Adaptive Moment Estimation)
Combines momentum and RMSprop:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 3. Evaluation Metrics Explained

1. **Precision**: $\frac{TP}{TP + FP}$ &mdash; Measures exactness (avoiding false alarms).
2. **Recall**: $\frac{TP}{TP + FN}$ &mdash; Measures completeness (catching every diseased leaf).
3. **F1-Score**: Harmonic mean: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
4. **Macro Average**: Unweighted mean across all classes, giving equal importance to rare and abundant classes.
5. **Weighted Average**: Averages per-class metrics weighted by sample support.

---

## 4. Key Viva Questions & Recommended Answers

| Question | Recommended Answer |
| :--- | :--- |
| **Why not train from scratch?** | Training from scratch on small datasets causes extreme overfitting. Pretrained weights on ImageNet provide universal low-level visual primitives (edges, contours, color gradients) that accelerate convergence. |
| **Why is Global Average Pooling used instead of Flatten?** | Flattening a $7 \times 7 \times 1280$ feature map to a dense layer creates millions of trainable parameters prone to overfitting. GAP averages each spatial map into a single scalar (1,280 features), dramatically reducing parameters while enforcing translational invariance. |
| **What does a high confidence score mean?** | High confidence means the input features closely align with the class clusters learned by the model. It does not calibrate against unseen novel pathogens or out-of-distribution inputs (e.g., non-leaf images). |
