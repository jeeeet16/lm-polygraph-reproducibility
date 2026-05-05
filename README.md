# Reproducibility: Evaluating Uncertainty Estimation Methods for Large Language Models using LM-Polygraph

This repository contains our reproducibility study based on the paper:

**LM-Polygraph: Uncertainty Estimation for Language Models** (Fadeeva et al., 2023)

We reproduce a subset of uncertainty estimation experiments and extend the analysis with:

- Cross-model evaluation (Vicuna-7B → Mistral-7B-Instruct-v0.2)
- Sampling parameter analysis (temperature, top-p, number of samples)

We expanded the original proposal's three methods to five — adding Perplexity and Lexical Similarity — to give a fuller picture across the white-box / black-box split.

---

## 📁 Project Structure

```
.
├── notebooks/
│   ├── 01_reproducibility_baseline.ipynb       # Vicuna-7B baseline
│   ├── 02_extension1_cross_model.ipynb         # Mistral-7B comparison
│   └── 03_extension2_sampling_sweep.ipynb      # Sampling parameter analysis
│
├── results/
│   ├── baseline_results.json
│   ├── baseline_prr_table.csv
│   ├── baseline_results.png
│   │
│   ├── extension1_results.json
│   ├── extension1_table.csv
│   ├── extension1_results.png
│   │
│   ├── extension2_results.json
│   ├── extension2_table.csv
│   └── extension2_results/
│       ├── temperature_sweep_results.png
│       ├── top_p_sweep_results.png
│       └── n_samples_sweep_results.png
│
├── src/
│   └── ue_repro_utils.py                       # Core utilities
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. HuggingFace login

Log in directly in the notebook with your access token — the HuggingFace popup login was unreliable in our setup:

```python
from huggingface_hub import login
login("YOUR_HF_TOKEN")
```

### 3. Run notebooks in order

```
notebooks/01_reproducibility_baseline.ipynb
notebooks/02_extension1_cross_model.ipynb
notebooks/03_extension2_sampling_sweep.ipynb
```

**GPU note:** A GPU is required for running Vicuna-7B and Mistral-7B efficiently. We used an A100 (Colab Pro) for the Baseline and Extension 1, and an NVIDIA T4 (AWS g4dn) for Extension 2.

---

## 📊 Experiments Overview

### 1. Baseline Reproduction (Vicuna-7B)

- **Dataset:** CoQA validation split (500 randomly sampled QA pairs, seed=42)
- **Model:** `lmsys/vicuna-7b-v1.5`
- **Methods:**
  - Maximum Sequence Probability (white-box)
  - Perplexity (white-box)
  - Mean Token Entropy (white-box)
  - Lexical Similarity (black-box)
  - Eccentricity (black-box)

**Outputs:**

```
results/baseline_results.json
results/baseline_prr_table.csv
results/baseline_results.png
```

**Goal:** Establish baseline uncertainty estimation performance and verify the relative ordering of methods reported in the original paper.

---

### 2. Extension 1: Cross-Model Evaluation (Mistral-7B)

- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Pipeline:** identical to baseline (same dataset subset, same five methods, same prompt)

**Outputs:**

```
results/extension1_results.json
results/extension1_table.csv
results/extension1_results.png
```

**Goal:** Test whether the relative performance of uncertainty estimation methods is consistent across different model architectures.

---

### 3. Extension 2: Sampling Parameter Analysis

- **Model:** Mistral-7B-Instruct-v0.2
- **Default sampling:** T=0.7, top-p=0.95, n_samples=5 (matches Extension 1)
- **Alternative settings tested:**
  - Temperature: 1.0
  - Top-p: 0.85
  - Number of samples: 10

White-box methods (Seq Prob, Perplexity, Token Entropy) are computed once from greedy decoding and reported as constants across the sweep. Black-box methods (Lexical Similarity, Eccentricity) are recomputed at each setting.

**Outputs:**

```
results/extension2_results.json
results/extension2_table.csv
results/extension2_results/temperature_sweep_results.png
results/extension2_results/top_p_sweep_results.png
results/extension2_results/n_samples_sweep_results.png
```

**Goal:** Analyze how decoding strategies affect uncertainty estimation, especially for sample-based black-box methods.

**Caveat:** Due to compute constraints, each sweep dimension is evaluated at a single alternative setting rather than a multi-point grid. The default-setting values (reported in `extension1_results.json`) serve as the implicit comparison point.

---

## 🧠 Methods Used

| Method                       | Type      | Category          |
| ---------------------------- | --------- | ----------------- |
| Maximum Sequence Probability | White-box | Information-based |
| Perplexity                   | White-box | Information-based |
| Mean Token Entropy           | White-box | Information-based |
| Lexical Similarity (ROUGE-L) | Black-box | Meaning-diversity |
| Eccentricity (NLI)           | Black-box | Meaning-diversity |

All methods are invoked via the official `lm-polygraph` library through estimator classes (`MaximumSequenceProbability`, `Perplexity`, `MeanTokenEntropy`, `LexicalSimilarity`, `Eccentricity`).

---

## 📌 Evaluation Metric

**Prediction Rejection Ratio (PRR)** measures how well an uncertainty score ranks incorrect predictions higher than correct ones:

$$\text{PRR} = \frac{\text{AUC}_{\text{model}} - \text{AUC}_{\text{random}}}{\text{AUC}_{\text{oracle}} - \text{AUC}_{\text{random}}}$$

Higher PRR means uncertainty estimates do a better job of identifying low-quality outputs. Correctness is assessed using token-level F1 against the CoQA reference, thresholded at 0.3 (binary correct/incorrect).

---

## ⚖️ Methodology Notes

A few intentional differences from the original paper that affect interpretability of absolute numbers:

- **Binary vs continuous PRR.** Our PRR uses _binary correctness_ (token-F1 ≥ 0.3) rather than the _continuous quality_ (ROUGE-L / BERTScore) used in Tables 2 and 3 of the original paper. This means our absolute PRR values are not directly comparable to the paper. Instead, we compare the **relative ordering** of methods, which is the more meaningful reproduction signal.
- **Subsampling.** Each experiment uses 500 randomly sampled CoQA QA pairs (seed=42), drawn across all stories rather than from a contiguous prefix.
- **Mistral prompt formatting.** We use the same plain prompt template across both models for an apples-to-apples pipeline comparison. Mistral-Instruct-v0.2 is chat-tuned and may underperform without `[INST]` formatting — this likely contributes to its lower mean F1 (0.161 vs Vicuna's 0.224).
- **4-bit quantization.** Both models are loaded with bitsandbytes 4-bit (NF4) quantization on GPU when available, which trades some generation quality for fitting into Colab/T4 VRAM.

---

## ⚙️ Utilities

Core logic lives in:

```
src/ue_repro_utils.py
```

Includes:

- `load_coqa(num_samples, seed)` — CoQA validation split sampling
- `token_f1(prediction, reference)` — SQuAD-style normalized token F1
- `compute_prr(uncertainty_scores, correctness)` — binary-correctness PRR
- `load_model(model_id, ...)` — HuggingFace + 4-bit quantization + LM-Polygraph `WhiteboxModel` wrapper
- `run_evaluation(...)` — end-to-end pipeline returning PRR per method
- `sweep_method_prrs(...)` — efficient parameter sweep that reuses greedy results across settings

---

## ⚠️ Notes

- Inference-only experiments (no training)
- Models loaded via HuggingFace Transformers
- Some models (e.g., Mistral-Instruct) require HuggingFace authentication
- Experiments run on a 500-example subset of CoQA due to compute limits

---

## 📎 Context

This repository supports a reproducibility report that:

- Reproduces a subset of LM-Polygraph experiments on CoQA
- Evaluates whether method rankings transfer across model architectures (Vicuna-7B → Mistral-7B)
- Studies the sensitivity of black-box uncertainty methods to decoding-time sampling parameters

## 📚 References

- Fadeeva et al. (2023). _LM-Polygraph: Uncertainty Estimation for Language Models._ arXiv:2311.07383
- Reddy, Chen, & Manning (2019). _CoQA: A Conversational Question Answering Challenge._ TACL
- Chiang et al. (2023). _Vicuna: An Open-Source Chatbot Impressing GPT-4._
- Jiang et al. (2023). _Mistral 7B._ arXiv:2310.06825
- Lin, Trivedi, & Sun (2023). _Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models._ arXiv:2305.19187
