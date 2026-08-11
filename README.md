# B.I.N.T.A.N.G: Automated Glaucoma Detection via Deep Ensemble Learning 👁️

This repository contains the deep learning codebase and presentation deck for **BINTANG (Biomedical Intelligence for Neural Tracking and Glaucoma)**, developed for the International Data Science Challenge (IDSC) 2026. The project secured the **1st Consolation Prize** for its scalable, safety-first approach to automated glaucoma screening.

## 📌 Clinical Problem
Every 60 seconds, one person in the world goes permanently blind, and 50% of glaucoma patients do not realize their condition until severe optic nerve damage occurs. Current screening systems struggle due to a lack of specialists and high variation in fundus image quality. BINTANG acts as a smart triage AI engine to empower primary healthcare workers with instant, explainable glaucoma risk scores.

## 📊 Dataset
Trained on the **Hillel-Yaffe Glaucoma Dataset (HYGD)** containing:
* **747** fundus images across **288** unique patients.
* Extreme class imbalance: 73.4% GON+ (Glaucoma) vs 26.6% GON- (Normal).

## 🛠️ AI Architecture & Methodology
* **4-Stage Clinical Preprocessing:** Implemented FOV Crop, Ben Graham Normalization, Green CLAHE, and Unsharp Masking to extract features from sub-optimal screening conditions.
* **Dual-Backbone Ensemble:** Fused **EfficientNet-B4** (for micro-precision) and **ConvNeXt-Base** (for global optic nerve context).
* **Quality-Weighted Focal Loss:** Designed a custom loss function that mathematically leverages per-image quality metadata to penalize hard borderline cases while counteracting the 73.4% class imbalance.
* **Uncertainty-Aware Triage:** Applied Multi-Scale Test Time Augmentation (TTA) using 135 augmentations per architecture to detect diagnostic ambiguity.
* **Glass-Box Explainability:** Integrated Grad-CAM to ensure the model focuses on the optic disc and genuine pathology rather than camera noise, ensuring regulatory readiness.

## 🚀 Performance & Clinical Impact
* **Image-Level Metrics:** Achieved an Out-of-Fold (OOF) AUC of **0.9888**.
* **Patient-Level Metrics:** By shifting the clinical threshold and applying Quality Score (QS)-weighted averaging per patient, the Patient-Level AUC reached **0.9950**.
* **Safety First:** The patient-level refinement drastically reduced False Negatives from 32 to 2 cases (a **93.75% reduction**), ensuring zero cross-patient contamination and prioritizing human sight over generic accuracy metrics.
## 📂 Repository Contents
* `BINTANG_Glaucoma_Detection.ipynb`: The end-to-end PyTorch training and inference pipeline.
* `BINTANG_Pitch_Deck.pdf`: The clinical presentation slides illustrating the pipeline, Grad-CAM results, and real-world impact.
