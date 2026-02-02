# From Supervised Fine-Tuning to PPO with Data Filtering

## Overview
This project studies the effect of **data filtering** and **reinforcement learning (PPO)** on fine-tuning a language model for generating short, engaging tweets.

Starting from a pretrained language model, we:
1. Perform **Supervised Fine-Tuning (SFT)** on a raw dataset.
2. **Filter the dataset** using model-based criteria.
3. Retrain the model using the **filtered dataset**.
4. Apply **Proximal Policy Optimization (PPO)** as a lightweight RLHF-style refinement.
5. Compare all resulting models **quantitatively and qualitatively**.

The main objective is to analyze how filtering and PPO influence:
- Output quality
- Stability
- Diversity
- Overfitting behavior

---

## Repository Structure
```bash
├── data/
│ ├── processed/ # Processed dataset (JSONL)
│ └── filtered/ # Filtered dataset
│
├── src/
│ ├── prepare_data.py # Dataset preprocessing & filtering
│ ├── sft_train.py # SFT on raw data
│ └── ppo_train.py # PPO training
│
├── raw_src/
│ └── sft_train_filtered.py # SFT on filtered data
│
├── outputs/
│ ├── sft/ # SFT model (raw dataset)
│ ├── sft_filtered/ # SFT model (filtered dataset)
│ ├── ppo/ # PPO model
│
├── outputs/logs/
│ ├── sft/ # TensorBoard logs (raw SFT)
│ ├── sft_filtered/ # TensorBoard logs (filtered SFT)
│ └── ppo/ # TensorBoard logs (PPO)
│ └── ppo_filtered/ # TensorBoard logs (PPO_filtered)
│
├── demo/
│ └── analysis.ipynb # Final analysis and comparison notebook
│
└── README.md
```

---

## Models

| Model | Description |
|------|-------------|
| Base | `distilgpt2` pretrained model |
| SFT (raw) | Fine-tuned on raw dataset |
| SFT (filtered) | Fine-tuned on filtered dataset |
| PPO | PPO-trained model initialized from SFT |

---

## Training Pipeline

### 1. Supervised Fine-Tuning (SFT – Raw)
- Dataset: processed raw dataset
- Objective: next-token prediction
- Metrics logged:
  - Training loss
  - Evaluation loss
  - Token-level accuracy

---

### 2. Dataset Filtering
- The trained SFT model is used to filter the dataset.
- Samples are selected based on quality-related heuristics.
- Result: a smaller but cleaner dataset.

---

### 3. Supervised Fine-Tuning (SFT – Filtered)
- Same training strategy as raw SFT.
- Observations:
  - Lower evaluation loss
  - More stable token accuracy
  - Reduced noise and overfitting risk

---

### 4. PPO Training
- PPO is applied as a reinforcement learning step on top of SFT.
- Rewards are derived from dataset metadata.
- Logged PPO metrics:
  - Reward mean
  - KL divergence
  - Entropy
  - Policy and value losses

---

## Evaluation

### Quantitative Metrics
- Training loss
- Evaluation loss
- Token accuracy (used as a proxy metric)
- PPO reward mean
- KL divergence and entropy

> Token accuracy is used for **relative comparison**, not as an absolute generation quality metric.

---

### Qualitative Analysis
- Fixed prompts are used to compare outputs from:
  - SFT (raw)
  - SFT (filtered)
  - PPO
- Results are presented in tables and discussed in detail.

---

##  Notebook

All analysis and comparisons are provided in: 


The notebook includes:
- Model loading
- Prompt-based comparisons
- Tables and plots
- Qualitative discussion

---

## 🧪 Key Findings

- Data filtering improves stability and reduces noise.
- Filtered SFT achieves lower evaluation loss than raw SFT.
- PPO can improve outputs in certain cases but is sensitive to reward design.
- PPO introduces a trade-off between reward optimization and output diversity.

---

## 🚀 How to Run

### Install dependencies
```bash
pip install torch transformers datasets trl accelerate numpy matplotlib
```
## Downlaod Data
```bash
chmod +x download_data.sh
./download_data.sh
```
## Run SFT on raw data
```bash
python src/sft_train.py
```
## Filter the dataset
```bash
python src/filter_data.py
```
## Run SFT on filtered data
```bash
python raw_src/sft_train_filtered.py
```
## Run PPO training
```bash
python src/ppo_train.py
```

