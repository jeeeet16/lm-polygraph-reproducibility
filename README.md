# Reproducibility: Evaluating Uncertainty Estimation Methods for Large Language Models using LM-Polygraph

This repository contains our reproducibility study based on the paper:

**LM-Polygraph: Uncertainty Estimation for Language Models**

We reproduce a subset of uncertainty estimation experiments and extend the analysis with:
- Cross-model evaluation (Vicuna-7B → Mistral-7B)
- Sampling parameter analysis (temperature, top-p, number of samples)

---

## 📁 Project Structure

```
.
├── notebooks/
│   ├── 01_reproducibility_baseline.ipynb
│   ├── 02_extension1_cross_model.ipynb
│   └── 03_extension2_sampling_sweep.ipynb
│
├── results/
│   ├── baseline_results.json
│   ├── baseline_prr_table.csv
│   ├── baseline_results.png
│
│   ├── extension1_results.json
│   ├── extension1_table.csv
│   ├── extension1_results.png
│
│   ├── extension2_results.json
│   ├── extension2_table.csv
│   └── extension2_results/
│       ├── temperature_sweep_results.png
│       ├── top_p_sweep_results.png
│       └── n_samples_sweep_results.png
│
├── src/
│   └── ue_repro_utils.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. HuggingFace Login

You will need to login using the following code

```
from huggingface_hub import login
login("YOUR_HF_TOKEN") 
```

Logging in using your token in the notebook will be easier, as huggingface did give us some issues while trying to login using its popup.

### 3. Run notebooks in order

```
notebooks/01_reproducibility_baseline.ipynb
notebooks/02_extension1_cross_model.ipynb
notebooks/03_extension2_sampling_sweep.ipynb
```

Please note that: GPU is required for running Vicuna-7B and Mistral-7B efficiently. We used A100 for Baseline and Extension 1, and used Nvidia G4 for Extension 2. 

---

## 📊 Experiments Overview

### 1. Baseline Reproduction (Vicuna-7B)

- Dataset: CoQA
- Model: Vicuna-7B
- Methods:
  - Maximum Sequence Probability
  - Perplexity
  - Mean Token Entropy
  - Lexical Similarity
  - Eccentricity

**Outputs:**
```
results/baseline_results.json
results/baseline_prr_table.csv
results/baseline_results.png
```

**Goal:** Establish baseline uncertainty estimation performance.

---

### 2. Extension 1: Cross-Model Evaluation (Mistral-7B)

- Model: Mistral-7B-Instruct-v0.2
- Same pipeline as baseline

**Outputs:**
```
results/extension1_results.json
results/extension1_table.csv
results/extension1_results.png
```

**Goal:** Test generalization of uncertainty methods across models.

---

### 3. Extension 2: Sampling Parameter Analysis

- Model: Mistral-7B
- Parameters tested:
  - Temperature
  - Top-p
  - Number of samples

**Outputs:**
```
results/extension2_results.json
results/extension2_table.csv
results/extension2_results/
```

**Goal:** Analyze how decoding strategies affect uncertainty estimation.

**Note:** Reduced parameter sweep due to compute constraints.

---

## 🧠 Methods Used

- Maximum Sequence Probability (white-box)
- Perplexity (white-box)
- Mean Token Entropy (white-box)
- Lexical Similarity (black-box)
- Eccentricity (black-box)

---

## 📌 Evaluation Metric

**Prediction Rejection Ratio (PRR)**

- Measures how well uncertainty identifies incorrect predictions
- Higher PRR = better uncertainty estimation

---

## ⚙️ Utilities

Core logic is implemented in:

```
src/ue_repro_utils.py
```

Includes:
- CoQA dataset loading
- Token-level F1 scoring
- PRR computation
- LM-Polygraph integration
- Evaluation pipeline

---

## ⚠️ Notes

- Inference-only experiments (no training)
- Models loaded via HuggingFace Transformers
- Some models require authentication
- Experiments run on a subset of CoQA due to compute limits

---

## 📎 Context

This repository supports a reproducibility report that:
- Reproduces LM-Polygraph experiments
- Evaluates robustness across models
- Studies sensitivity to decoding strategies