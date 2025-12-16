# Automated Heartbeat Sound Classification and Anomaly Detection  
### Summer Internship Project (SIP), NIT Durgapur

This repository contains the work carried out during my **Summer Research Internship at NIT Durgapur**, focused on developing a **biomedical signal processing and machine learning pipeline** for automated heartbeat sound analysis.

The project integrates:
- **Biomedical signal processing of phonocardiogram (PCG) signals**
- **Feature-based machine learning with Bayesian optimization**
- **Semi-supervised anomaly detection using deep learning**

---

## Dataset and Feature Representation

- Heartbeat sound recordings were segmented into individual cycles
- Each cycle was represented using MFCC-based feature vectors
  
---
## Methodology

### Biomedical Signal Processing Pipeline

Heartbeat sounds are non-stationary biomedical signals with strong physiological structure.  
To ensure clinically meaningful analysis, a dedicated **signal processing pipeline** was designed prior to model training.

- Preprocessing of raw PCG audio signals using **Bandpass Butterworth Filter**  
- Normalization and noise handling  
- Segmentation of recordings into individual cardiac cycles using **Hilbert Transform**
- Extraction of **MFCC (Mel-Frequency Cepstral Coefficients)** to capture spectral characteristics of heart sounds  
- Preprocessed features and labels were stored as NumPy arrays (`X.npy`, `y.npy`) for reproducibility and efficient experimentation

---

### Supervised Classification

The first phase addressed **heartbeat sound classification** using labeled data.

**Models used:**
- **Support Vector Machine (SVM)** – margin-based classifier suitable for high-dimensional MFCC features  
- **Random Forest (RF)** – ensemble of decision trees providing robustness to noise  
- **XGBoost (XGB)** – boosting-based classifier capturing complex feature interactions  

#### Bayesian Hyperparameter Optimization (SMAC)

To avoid ad-hoc tuning, **SMAC (Sequential Model-based Algorithm Configuration)** was used for **Bayesian hyperparameter optimization**, enabling principled and reproducible model selection.

SMAC optimized:
- SVM (kernel, C, gamma)  
- Random Forest (number of trees, depth)  
- XGBoost (learning rate, tree depth, estimators)  

---

### Anomaly Detection (Semi-Supervised)

To reflect realistic clinical deployment, the problem was reformulated as **anomaly detection**, where models are trained exclusively on **normal heartbeats**.

**Anomaly detection methods implemented:**

| Method | Description |
|------|-------------|
| **One-Class SVM** | Learns a boundary around normal cardiac cycles |
| **Isolation Forest** | Ensemble-based anomaly detection via random isolation |
| **Dense Bottleneck Autoencoder** | Learns compact representations of normal heart sounds and detects anomalies using reconstruction error |

The autoencoder was implemented using **TensorFlow/Keras**, introducing deep learning–based representation learning into the pipeline.

---
