# 🧠 LiteGAN-FedNet: An Explainable Federated Learning Framework with Conditional GAN Feature Augmentation for Brain Tumor MRI Classification

> **Final Year Undergraduate Thesis**
>
> **Authors:** MD Irfanul Kabir Hira · Mst Moriom Akter Bithee · MD Sohag Hossain · Umme Sara · MD Mahmudul Hasan · Abdullah Al Towsif · Md Kowsar Ahmed

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Research Objectives](#-research-objectives)
3. [Pipeline Architecture](#-pipeline-architecture)
4. [Datasets](#-datasets)
5. [Tumor Classes](#-tumor-classes)
6. [Project Steps](#-project-steps)
   - [Step 1 — Upload & Extract Dataset](#step-1--upload--extract-dataset)
   - [Step 2 — Define ResNet18 Model](#step-2--define-resnet18-model)
   - [Step 3 — Train ResNet18](#step-3--train-resnet18)
   - [Step 4 — Feature Extraction](#step-4--feature-extraction)
   - [Step 5 — Conditional GAN (cGAN) Training](#step-5--conditional-gan-cgan-training)
   - [Step 6 — Build Federated Client Datasets](#step-6--build-federated-client-datasets)
   - [Step 7 — Synthetic Feature Generation & Merging](#step-7--synthetic-feature-generation--merging)
   - [Step 8 — Aggregation, Scaling & PCA](#step-8--aggregation-scaling--pca)
   - [Step 9 — Train MLP Classifier](#step-9--train-mlp-classifier)
   - [Step 10 — SHAP Explainability](#step-10--shap-explainability)
   - [Step 11 — Track Training Loss & Validation Accuracy](#step-11--track-training-loss--validation-accuracy)
   - [Step 12 — Plot Loss & Accuracy Curves](#step-12--plot-loss--accuracy-curves)
   - [Step 13 — Grad-CAM Visualization](#step-13--grad-cam-visualization)
   - [Step 14 — SHAP on PCA Features (Final)](#step-14--shap-on-pca-features-final)
   - [Step 15 — LIME Implementation](#step-15--lime-implementation)
   - [Confusion Matrix](#confusion-matrix)
7. [Installation & Setup](#-installation--setup)
8. [How to Run](#-how-to-run)
9. [Key Results](#-key-results)
10. [Technologies Used](#-technologies-used)
11. [Datasets Citation](#-datasets-citation)

---

## 🔍 Project Overview

This is the final year undergraduate thesis project implementing a **hybrid deep learning pipeline** for **Brain Tumor MRI Classification** that combines:

- **Federated Learning (FL)** — Simulates 3 hospital clients, each training locally without sharing raw MRI data, preserving patient privacy
- **Conditional GAN (cGAN)** — Generates synthetic feature vectors per class to balance the dataset across clients
- **ResNet18** — Pre-trained CNN used as a feature extractor (512-dimensional features)
- **PCA** — Reduces 512D features to 100D for efficient learning
- **MLP Classifier** — Final classification head on aggregated PCA features
- **Explainable AI (XAI)** — SHAP, Grad-CAM, and LIME used to interpret model decisions

The system classifies brain MRI scans into **4 categories**: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

---

## 🎯 Research Objectives

1. Build a **privacy-preserving** brain tumor classification system using Federated Learning
2. Address **class imbalance** across federated clients using Conditional GAN synthetic feature generation
3. Achieve **efficient training** through PCA dimensionality reduction (512D → 100D)
4. Provide **explainable predictions** via SHAP (global + local), Grad-CAM (visual saliency), and LIME

---

## 🔄 Pipeline Architecture

```
MRI Images (Mendeley + Figshare Datasets)
            │
            ▼
  Data Augmentation & Preprocessing
            │
            ▼
  ResNet18 (Pre-trained, ImageNet)
  → Fine-tuned on Brain MRI (4 classes)
            │
            ▼
  Feature Extraction (512-D per image)
            │
       ┌────┴────────────────────┐
       ▼                         ▼
Conditional GAN           Split into 3 FL Clients
(Synthetic Feature        (simulating 3 hospitals)
 Generation per class)          │
       │                         ▼
       └──── Merge Real + Synthetic Features (per client)
                         │
                         ▼
              Aggregate All Client Features
                         │
                         ▼
              StandardScaler + PCA (512D → 100D)
                         │
                         ▼
                  MLP Classifier
                         │
                         ▼
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
       SHAP           Grad-CAM          LIME
  (Feature-level)  (Pixel-level)  (Superpixel-level)
```

---

## 📊 Datasets

Two publicly available Brain MRI datasets are used:

### 1. Mendeley Brain Tumor MRI Dataset
| Property | Details |
|---|---|
| **URL** | https://data.mendeley.com/datasets/zwr4ntf94j/5 |
| **DOI** | 10.17632/zwr4ntf94j.5 |
| **Version** | V5 (2025) |
| **Classes** | Glioma, Meningioma, Pituitary, No Tumor |

### 2. Figshare Brain MRI Dataset
| Property | Details |
|---|---|
| **URL** | https://figshare.com/articles/dataset/Brain_MRI_Dataset/14778750 |
| **DOI** | 10.6084/m9.figshare.14778750 |

> ⚠️ Datasets are **not included** in this repository. Please download them directly from the links above and upload as ZIP files when running the notebook.

---

## 🧬 Tumor Classes

| Class | Description |
|---|---|
| **Glioma** | Malignant tumor arising from glial cells |
| **Meningioma** | Tumor arising from the meninges (usually benign) |
| **Pituitary** | Tumor in the pituitary gland |
| **No Tumor** | Healthy brain MRI (control class) |

---

## 📁 Project Steps

### Step 1 — Upload & Extract Dataset
Accepts two ZIP files (Mendeley and Figshare datasets) uploaded via Google Colab's file uploader. Extracts them to separate local folders and inspects the directory structure to verify class subfolders.

### Step 2 — Define ResNet18 Model
Loads pre-trained **ResNet18** (ImageNet weights) from `torchvision.models`. Replaces the final fully connected layer with a custom `nn.Linear(512, 4)` output head for 4-class brain tumor classification. Sets up GPU/CPU device.

### Step 3 — Train ResNet18
Trains ResNet18 with:
- **Optimizer:** Adam (lr = 1e-4)
- **Loss:** CrossEntropyLoss
- **Epochs:** 10
- **Batch Size:** 16
- **Augmentation:** Horizontal flip, rotation (±15°), affine translation, color jitter, random crop

Prints train loss, train accuracy, val loss, and val accuracy per epoch.

### Step 4 — Feature Extraction
Removes the final `fc` layer from the trained ResNet18 to create a **feature extractor**. Passes all training and test images through this truncated network to get **512-dimensional feature vectors** per image. Outputs shape: `(N, 512)`.

### Step 5 — Conditional GAN (cGAN) Training
Trains a **Conditional GAN** on the extracted 512-D features:
- **Generator:** Takes noise vector (128-D) + class label embedding → produces 512-D synthetic feature
- **Discriminator:** Takes 512-D feature + class label → outputs real/fake probability
- **Purpose:** Generate balanced synthetic features for under-represented tumor classes across FL clients

### Step 6 — Build Federated Client Datasets
Splits the full dataset across **3 federated clients** (simulating 3 hospital nodes). Each client receives an approximately equal partition of the real feature data. Prints per-client sizes to verify.

### Step 7 — Synthetic Feature Generation & Merging
For each of the 3 clients:
1. Collects the client's real ResNet18 features
2. Uses the trained Generator to produce `200 synthetic samples per class`
3. Concatenates real + synthetic → **augmented, balanced client dataset**

### Step 8 — Aggregation, Scaling & PCA
- **Aggregation:** Combines all 3 client feature sets into one global dataset
- **StandardScaler:** Fits on training features only; transforms train and test
- **PCA:** Reduces 512D → **100D** (fitted on training set only, applied to test set)
- Preserves ~95%+ variance while making training much more efficient

### Step 9 — Train MLP Classifier
Trains a multi-layer perceptron on the 100-D PCA features:
- Input: 100-D PCA features
- Hidden layers with ReLU activations and dropout
- Output: 4-class softmax
- Loss: CrossEntropyLoss | Optimizer: Adam

### Step 10 — SHAP Explainability
Computes **SHAP values** on the MLP classifier:
- **Summary (beeswarm) plot** — shows which PCA components drive predictions
- **Bar chart** — global feature importance ranking
- Uses `shap.DeepExplainer` or `shap.KernelExplainer` with Seaborn-styled visualization

### Step 11 — Track Training Loss & Validation Accuracy
Records and stores epoch-wise training loss and validation accuracy for the MLP (FL + GAN + PCA stage). Used for convergence analysis and comparison.

### Step 12 — Plot Loss & Accuracy Curves
Generates clean Seaborn-styled plots of:
- Training loss vs epoch
- Validation accuracy vs epoch

Useful for diagnosing overfitting and demonstrating model convergence.

### Step 13 — Grad-CAM Visualization
Implements **Gradient-weighted Class Activation Mapping (Grad-CAM)** on the ResNet18 model. Produces pixel-level **heatmaps** overlaid on original MRI images, highlighting which brain regions the model focuses on for each tumor class prediction.

### Step 14 — SHAP on PCA Features (Final)
Final clean SHAP analysis run on PCA features with polished Seaborn visualizations. Provides the publication-ready global and local explainability plots.

### Step 15 — LIME Implementation
Implements **Local Interpretable Model-agnostic Explanations (LIME)** on individual MRI image predictions. Highlights **superpixel regions** that positively or negatively contributed to each classification decision.

### Confusion Matrix
Plots the **confusion matrix** for the MLP classifier across all 4 tumor classes, showing per-class precision, recall, and misclassification patterns.

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended — tested with CUDA 12.6)
- Google Colab (recommended) or local Jupyter environment

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib pillow tqdm opencv-python pandas
pip install flwr
pip install shap
pip install albumentations xgboost
```

Or run the first cell of the notebook which installs everything automatically.

### Verified Versions
| Package | Version |
|---|---|
| PyTorch | 2.8.0 + CUDA 12.6 |
| Flower (flwr) | 1.22.0 |
| SHAP | 0.48.0 |
| Albumentations | 2.0.8 |
| XGBoost | 3.0.5 |

---

## ▶️ How to Run

1. **Download the datasets** from the links in the [Datasets](#-datasets) section and prepare them as ZIP files.

2. **Open the notebook** — Upload `training_mendeley.ipynb` to Google Colab.

3. **Run all cells** — `Runtime → Run all`

4. **Upload datasets** — When prompted in Step 1, upload your two dataset ZIP files (Mendeley + Figshare).

5. **GPU recommended** — Go to `Runtime → Change runtime type → GPU` for faster training.

---

## 📈 Key Results

- **Hybrid pipeline** (FL + cGAN + PCA + MLP) achieves competitive classification across 4 brain tumor classes
- **Federated setup** across 3 clients preserves data privacy while maintaining model utility
- **cGAN** effectively balances class distribution within each client's local dataset
- **PCA** reduces training time significantly (512D → 100D) with minimal accuracy loss
- **SHAP + Grad-CAM + LIME** together provide multi-level interpretability (feature, pixel, superpixel)

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch, TorchVision |
| **Pre-trained Model** | ResNet18 (ImageNet) |
| **Federated Learning** | Flower (flwr) |
| **Generative Model** | Conditional GAN (custom PyTorch) |
| **Dimensionality Reduction** | PCA (scikit-learn) |
| **Explainability** | SHAP, Grad-CAM, LIME |
| **Data Processing** | pandas, NumPy, OpenCV, Albumentations |
| **Visualization** | matplotlib, seaborn |
| **Environment** | Google Colab (GPU) / Jupyter Notebook |

---

## 📚 Datasets Citation

```bibtex
@data{zwr4ntf94j,
  author    = {HIRA, MD IRFANUL KABIR and MST MORIOM AKTER  and HOSSAIN, MD SOHAG
               and Sara, Umme Sara and HASAN, MD MAHMUDUL and Towsif, Abdullah Al
               and Ahmed, Md Kowsar},
  title     = {Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)},
  year      = {2025},
  note      = {Under Review}
}
```



---

*Final Year Undergraduate Thesis — Brain Tumor Classification with Privacy-Preserving Federated Learning and Explainable AI*
