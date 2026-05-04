# **LM-Polygraph: Uncertainty Estimation for Language Models**

The project reproduces selected uncertainty estimation experiments using the LM-Polygraph framework and extends the analysis with cross-model evaluation and sampling-parameter experiments.

## Project Structure

```text
.
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
